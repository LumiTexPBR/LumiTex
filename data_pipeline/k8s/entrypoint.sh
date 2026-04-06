#!/bin/bash
set -euo pipefail

# ── Required environment variables (from ConfigMap / K8s Job) ────────
: "${JOB_COMPLETION_INDEX:?JOB_COMPLETION_INDEX not set}"
: "${TOTAL_UIDS:?TOTAL_UIDS not set}"
: "${CHUNK_SIZE:?CHUNK_SIZE not set}"

# ── Calculate UID slice for this pod ─────────────────────────────────
START_INDEX=$((JOB_COMPLETION_INDEX * CHUNK_SIZE))
END_INDEX=$((START_INDEX + CHUNK_SIZE))
if [ "${END_INDEX}" -gt "${TOTAL_UIDS}" ]; then
    END_INDEX=${TOTAL_UIDS}
fi

if [ "${START_INDEX}" -ge "${TOTAL_UIDS}" ]; then
    echo "[entrypoint] START_INDEX (${START_INDEX}) >= TOTAL_UIDS (${TOTAL_UIDS}), nothing to do."
    exit 0
fi

# ── Ensure output directories exist ──────────────────────────────────
mkdir -p /data/rendered_uids

echo "============================================="
echo " Pod index:    ${JOB_COMPLETION_INDEX}"
echo " UID range:    [${START_INDEX}, ${END_INDEX})"
echo " Processes:    ${NUM_PROCESSES:-4}"
echo " GPU ID:       0 (K8s isolated)"
echo "============================================="

# ── Optional: configure WandB ────────────────────────────────────────
WANDB_ARGS=""
if [ "${LOG_TO_WANDB:-false}" = "true" ]; then
    WANDB_ARGS="--log_to_wandb"
fi

# ── Optional: render material flag ───────────────────────────────────
MATERIAL_ARGS=""
if [ "${RENDER_MATERIAL:-true}" = "true" ]; then
    MATERIAL_ARGS="--render_material"
fi

# ── Run the renderer ─────────────────────────────────────────────────
exec python3 render_objaverse.py \
    --download_dir   /data/objaverse \
    --envmap_dir     /data/envmaps \
    --data_uids      "${DATA_UIDS_PATH:-/data/uids/data_uids.json}" \
    --rendered_uids_txt "/data/rendered_uids/rendered_uids_${JOB_COMPLETION_INDEX}.txt" \
    --output_dir     /data/output \
    --start_index    "${START_INDEX}" \
    --end_index      "${END_INDEX}" \
    --processes      "${NUM_PROCESSES:-4}" \
    --gpu_id         0 \
    --s3_config      "${S3_CONFIG_PATH:-/etc/s3/s3_config.yaml}" \
    ${WANDB_ARGS} \
    ${MATERIAL_ARGS}
