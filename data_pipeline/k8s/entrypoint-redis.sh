#!/bin/bash
set -euo pipefail

echo "============================================="
echo " LumiTex Redis Worker"
echo " Host:    $(hostname)"
echo " Redis:   ${REDIS_HOST:-lumitex-redis}:${REDIS_PORT:-6666}"
echo " Prefix:  ${REDIS_KEY_PREFIX:-lumitex}"
echo "============================================="

# Ensure output directories exist
mkdir -p /data/rendered_uids

exec python3 /app/data_pipeline/k8s/redis_worker.py
