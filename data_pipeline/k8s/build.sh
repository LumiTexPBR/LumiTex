#!/bin/bash
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────
REGISTRY="${REGISTRY:-docker.io/lumitex}"
TAG="${TAG:-latest}"
PUSH=false

# ── Parse args ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY="$2"; shift 2 ;;
        --tag)      TAG="$2";      shift 2 ;;
        --push)     PUSH=true;     shift   ;;
        -h|--help)
            echo "Usage: $0 [--registry REGISTRY] [--tag TAG] [--push]"
            echo ""
            echo "  --registry   Container registry (default: docker.io/lumitex)"
            echo "  --tag        Image tag (default: latest)"
            echo "  --push       Push image after build"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

IMAGE="${REGISTRY}/lumitex-renderer:${TAG}"

# ── Build context must be the project root ───────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Building image: ${IMAGE}"
echo "Build context:  ${PROJECT_ROOT}"
echo "Dockerfile:     ${SCRIPT_DIR}/Dockerfile"

docker build \
    -t "${IMAGE}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${PROJECT_ROOT}"

echo "Build complete: ${IMAGE}"

if [ "${PUSH}" = true ]; then
    echo "Pushing ${IMAGE} ..."
    docker push "${IMAGE}"
    echo "Push complete."
fi
