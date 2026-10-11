"""Live end-to-end check: prompt -> spec -> dataset plan -> approved build -> files.

This drives the real HTTP API and the real worker over the real network and then checks
the delivered artifacts on disk, so it is the closest thing to the demo that a script
can assert.

It is deliberately **not** part of the test suite: it needs the Hub, the planner LLM and
a couple of minutes (it downloads the real dataset). Run it after changing Stage 1 or 2,
with the API and a worker up:

    python -m slmforge serve      # terminal 1
    python -m slmforge worker     # terminal 2
    python scripts/demo_smoke.py  # terminal 3

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import httpx

API = "http://127.0.0.1:8000/api"
PROMPT = (
    "Build a small model that classifies customer support tickets into Billing, "
    "Technical, Account Access or General and returns only a valid JSON object."
)
ARTIFACT_ROOT = pathlib.Path("artifacts")

failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def main() -> int:
    client = httpx.Client(base_url=API, timeout=120.0)

    caps = client.get("/capabilities").json()
    print(
        f"capabilities: sandbox={caps['sandbox_backend']} docker_error={caps.get('sandbox_error')}"
    )

    project = client.post("/projects", json={"prompt": PROMPT}).json()
    pid = project["id"]
    print(f"\nproject {pid}")

    print("\n[1] analyze (planner LLM, with a deterministic fallback)")
    body = client.post(f"/projects/{pid}/stages/analyze/run", json={}).json()
    spec = body["output"]
    print(f"  status={body['status']} via={spec.get('generated_by')} readiness={spec['readiness']}")
    print(f"  labels={spec['allowed_labels']} format={spec['output']['format']}")
    check(body["status"] == "succeeded", "analyzer succeeds")
    check(
        spec["allowed_labels"] == ["Billing", "Technical", "Account Access", "General"],
        "analyzer keeps the user's four labels",
        str(spec["allowed_labels"]),
    )
    check(spec["readiness"] == "ready_for_review", "spec is reviewable, not blocked")

    print("\n[2] curate_data plan (live remote profile, nothing downloaded)")
    body = client.post(f"/projects/{pid}/stages/curate_data/run", json={}).json()
    plan = body["output"]
    print(f"  status={body['status']} needs_approval={body['needs_approval']} {body['blocked_on']}")
    print(f"  summary: {body['summary']}")
    for candidate in plan["candidates"]:
        print(f"  - {candidate['id']} fit={candidate['fit_score']} rows={candidate['rows']}")
        print(f"      coverage={candidate['label_coverage']}")
        for warning in candidate["warnings"]:
            print(f"      ! {warning}")
    check(body["status"] == "succeeded", "plan succeeds")
    check(body["needs_approval"] is True, "plan stops at the approval gate")
    check(
        client.get(f"/projects/{pid}").json()["status"] == "specified",
        "the project does not advance to data_ready on a plan alone",
    )
    recommended = plan["candidates"][0]["id"]
    coverage = plan["candidates"][0]["label_coverage"]
    check(coverage.get("Technical", 0) > 0, "live profile produced real label counts")
    check(
        any("Account Access" in w for w in plan["candidates"][0]["warnings"]),
        "the empty class is reported, not hidden",
    )

    print("\n[3] approve: enqueue the build as a job (nothing downloads in-request)")
    inline = client.post(
        f"/projects/{pid}/stages/curate_data/run", json={"params": {"build": True}}
    )
    check(inline.status_code == 409, "a synchronous build is refused", str(inline.status_code))

    queued = client.post(
        f"/projects/{pid}/jobs",
        json={
            "kind": "dataset_build",
            "params": {"build": True, "selected": [recommended], "target_examples": 4000},
        },
    )
    check(queued.status_code == 201, "build job enqueued", queued.text[:200])
    job_id = queued.json()["id"]

    print(f"\n[4] waiting for worker to run job {job_id[:8]} ...")
    deadline = time.time() + 900
    job: dict = {}
    while time.time() < deadline:
        job = next(j for j in client.get(f"/projects/{pid}/jobs").json() if j["id"] == job_id)
        if job["status"] not in {"pending", "running"}:
            break
        time.sleep(5)
    print(
        f"  job status={job.get('status')} attempts={job.get('attempts')} error={job.get('error')}"
    )
    check(job.get("status") == "succeeded", "the queued build succeeded")

    print("\n[5] verify what was actually delivered")
    detail = client.get(f"/projects/{pid}").json()
    print(f"  project status={detail['status']}")
    card = (detail.get("dataset_plan") or {}).get("card")
    check(detail["status"] == "data_ready", "the project advanced after a real build")
    if not card:
        check(False, "dataset card present", "no card on the project")
        return _report(client, pid)

    print(f"  total={card['total_examples']} splits={card['splits']}")
    print(f"  labels={card['label_distribution']}")
    print(f"  synthetic_fraction={card['synthetic_fraction']}")
    print(f"  sufficiency={(detail['dataset_plan']).get('sufficiency')}")
    for source in card["sources"]:
        print(
            f"  source={source['source']} uri={source['uri']} licence={(source['license'] or {}).get('id')}"
        )
    for issue in card["issues"]:
        print(
            f"  issue[{issue['severity']}] {issue['kind']}: {issue['count']} -- {issue['detail']}"
        )

    check(card["total_examples"] > 0, "split files contain examples")
    check(
        card["splits"]["train"] + card["splits"]["validation"] + card["splits"]["test"]
        == card["total_examples"],
        "every example is in exactly one split",
    )
    check(card["synthetic_fraction"] > 0, "the synthetic share is recorded, not hidden")
    check(
        any(issue["kind"] == "missing_class" for issue in card["issues"]),
        "the unlearnable class is reported as an issue",
    )

    project_dir = ARTIFACT_ROOT / pid / "dataset"
    for name in ("train", "validation", "test"):
        path = project_dir / f"{name}.jsonl"
        exists = path.is_file()
        check(exists, f"{name}.jsonl written", str(path))
        if not exists:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        check(digest == card["sha256"][name], f"{name}.jsonl hash matches the card")
        lines = path.read_text(encoding="utf-8").splitlines()
        check(len(lines) == card["splits"][name], f"{name}.jsonl has the recorded row count")
        sample = json.loads(lines[0])
        check(
            [m["role"] for m in sample["messages"]] == ["system", "user", "assistant"],
            f"{name}.jsonl is chat format",
        )

    # Artifacts accumulate per project (the spec from Stage 1, the splits from Stage
    # 2), so the dataset files must be a subset here, not the whole set.
    recorded = {a["name"] for a in detail["artifacts"]}
    expected = {"train.jsonl", "validation.jsonl", "test.jsonl", "dataset_card.json"}
    check(
        expected <= recorded,
        "dataset artifacts are registered on the project",
        str(sorted(recorded)),
    )
    check(
        card["total_examples"] <= 4000,
        "the build respects the approved example count",
        f"delivered {card['total_examples']} for target 4000",
    )
    check(
        card["total_examples"] == sum(card["splits"].values()),
        "the card adds up: total equals the split sizes",
    )
    check(
        sum(card["label_distribution"].values()) == card["total_examples"],
        "the card adds up: labels equal the total",
    )
    return _report(client, pid)


def _report(client: httpx.Client, pid: str) -> int:
    print("\n[6] event log (what the dashboard will show)")
    for event in reversed(client.get(f"/projects/{pid}/events").json()):
        print(f"  {event['level']:7} {event['stage'] or '-':11} {event['message'][:130]}")
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
