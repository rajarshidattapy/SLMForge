# SLMForge sandbox image — untrusted dataset prep and CPU evaluation.
#
# Deliberately small and deliberately missing the training stack: heavy work
# belongs on tier 2 (managed GPU containers) or a separately built training
# image. A slim, auditable base means a smaller attack surface and a faster pull.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# duckdb   : dataset profiling over remote parquet
# pandas   : normalization
# pyarrow  : parquet read/write
# sklearn  : label encoding / metrics used by the evaluation harness
RUN pip install --no-cache-dir \
      "duckdb==1.5.6" \
      "pandas==2.3.3" \
      "pyarrow>=17" \
      "scikit-learn>=1.5" \
      "huggingface_hub==1.8.0"

# /work is bind-mounted per job by DockerSandbox; the directory here only exists
# so the image is usable standalone for debugging.
WORKDIR /work

# Non-root by default as well, so the boundary holds even if `--user` is omitted.
USER 65534:65534
