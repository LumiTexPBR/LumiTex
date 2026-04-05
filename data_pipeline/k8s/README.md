# LumiTex K8s Rendering Pipeline

Kubernetes deployment for distributed Blender rendering of the Objaverse dataset.

## Prerequisites

- Kubernetes cluster with **NVIDIA GPU Operator** and `nvidia-device-plugin`
- StorageClass supporting **ReadWriteMany** (e.g. NFS, CephFS)
- `kubectl` configured with cluster access
- Docker (for building the image)

## Quick Start

### 1. Build & push the container image

```bash
# Build locally
bash build.sh --registry your-registry.com/lumitex --tag v1

# Build and push
bash build.sh --registry your-registry.com/lumitex --tag v1 --push
```

### 2. Create namespace

```bash
kubectl create namespace lumitex
```

### 3. Configure secrets

```bash
# Copy template and fill in real credentials
cp secret.yaml secret-local.yaml
# Edit secret-local.yaml with real bucket, keys, etc.
kubectl apply -f secret-local.yaml
```

### 4. Pre-load data onto PVC

```bash
# Create PVC (edit storageClassName first if needed)
kubectl apply -f pvc.yaml

# Use a temporary pod to copy envmaps and UID files onto the PVC:
kubectl run data-loader --image=busybox --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"loader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"lumitex-shared-data"}}]}}' \
  -n lumitex

kubectl cp ./envmaps lumitex/data-loader:/data/envmaps
kubectl cp ./data/data_uids.json lumitex/data-loader:/data/uids/data_uids.json
kubectl delete pod data-loader -n lumitex
```

### 5. Deploy the rendering job

```bash
# Review and apply ConfigMap
kubectl apply -f configmap.yaml

# Update image reference in job.yaml, then:
kubectl apply -f job.yaml
```

## Configuration

### ConfigMap (`configmap.yaml`)

| Key | Description | Default |
|-----|-------------|---------|
| `TOTAL_UIDS` | Total UIDs in dataset | `63000` |
| `CHUNK_SIZE` | UIDs per pod | `1000` |
| `NUM_PROCESSES` | Blender workers per pod | `4` |
| `RENDER_MATERIAL` | Enable PBR material rendering | `true` |
| `LOG_TO_WANDB` | Enable WandB logging | `false` |

### Job (`job.yaml`)

| Field | Description | Default |
|-------|-------------|---------|
| `completions` | Total pods = ceil(TOTAL_UIDS/CHUNK_SIZE) | `63` |
| `parallelism` | Max concurrent pods | `8` |
| `resources.requests.nvidia.com/gpu` | GPUs per pod | `1` |
| `resources.requests.cpu` | CPU per pod | `4` |
| `resources.requests.memory` | Memory per pod | `16Gi` |

## Monitoring

```bash
# Job status
kubectl get job lumitex-render -n lumitex

# List pods
kubectl get pods -n lumitex -l app=lumitex-renderer

# Logs for a specific pod
kubectl logs lumitex-render-<INDEX> -n lumitex

# Watch progress
kubectl get pods -n lumitex -l app=lumitex-renderer -w
```

If `LOG_TO_WANDB=true`, check the WandB dashboard (project: `ObjaverseRenderer-LumiTex`).

## Architecture

```
K8s Indexed Job (completions=63, parallelism=8)
  |
  +-- Pod 0  (INDEX=0)  -> render UIDs [0, 1000)
  +-- Pod 1  (INDEX=1)  -> render UIDs [1000, 2000)
  +-- ...
  +-- Pod 62 (INDEX=62) -> render UIDs [62000, 63000)

Each pod:
  entrypoint.sh -> render_objaverse.py
    -> multiprocessing.Pool(4)
    -> blender -b -P blender_script.py (CUDA CYCLES)
    -> validate 288 images per object
    -> zip + upload to S3/COS + cleanup
```

## Troubleshooting

**Pod stuck in Pending:**
- Check GPU availability: `kubectl describe nodes | grep nvidia.com/gpu`
- Verify NVIDIA device plugin: `kubectl get pods -n kube-system | grep nvidia`

**OOMKilled:**
- Increase `resources.limits.memory` in `job.yaml`
- Reduce `NUM_PROCESSES` in `configmap.yaml`

**S3 auth failures:**
- Verify credentials in secret: `kubectl get secret lumitex-s3-credentials -n lumitex -o yaml`
- Ensure `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are correct for Tencent COS

**Blender CUDA errors:**
- Ensure GPU driver is compatible with CUDA 12.4
- Check NVIDIA runtime: `kubectl logs <pod> | grep -i cuda`
