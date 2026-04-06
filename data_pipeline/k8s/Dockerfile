FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BLENDER_URL=https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz

# ── System dependencies + Python 3.11 via deadsnakes PPA ────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        p7zip-full wget xz-utils ca-certificates \
        # Blender runtime libs
        libgl1 libglu1-mesa libxi6 libxrender1 libxxf86vm1 \
        libxkbcommon0 libsm6 libxext6 libxfixes3 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ── Install pip for Python 3.11 ─────────────────────────────────────
RUN python3 -m ensurepip --upgrade \
    && python3 -m pip install --no-cache-dir --upgrade pip

# ── Install Blender 4.2 LTS ─────────────────────────────────────────
RUN wget -qO /tmp/blender.tar.xz ${BLENDER_URL} \
    && mkdir -p /opt/blender \
    && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz

ENV PATH="/opt/blender:${PATH}"

# ── Python dependencies (system Python for render_objaverse.py) ─────
RUN python3 -m pip install --no-cache-dir \
        numpy objaverse boto3 OpenEXR Pillow tqdm wandb pyyaml redis

# ── Python dependencies (Blender's bundled Python for blender_script.py)
# ensurepip may be stripped from Blender's bundled Python; fall back to get-pip.py
RUN /opt/blender/4.2/python/bin/python3.11 -m ensurepip 2>/dev/null \
    || (wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
        && /opt/blender/4.2/python/bin/python3.11 /tmp/get-pip.py \
        && rm /tmp/get-pip.py) \
    && /opt/blender/4.2/python/bin/python3.11 -m pip install --no-cache-dir \
        numpy Pillow

# ── Copy project code ───────────────────────────────────────────────
COPY utils/          /app/utils/
COPY data_pipeline/  /app/data_pipeline/

WORKDIR /app/data_pipeline

# Symlink so bare "blender_script.py" in the blender command resolves
RUN ln -sf blender_scripts/blender_script.py blender_script.py

# PYTHONPATH:
#   /app                                 -> "from utils import *"  (render_objaverse.py, blender_utils_helper.py)
#   /app/data_pipeline                   -> "from file_checker import *", "from parse_exr_depth import *"
#   /app/data_pipeline/blender_scripts   -> "from blender_utils_helper import *"  (blender_script.py)
#     ^ Required because the symlink makes os.path.abspath(__file__) resolve to
#       /app/data_pipeline/ instead of blender_scripts/, so the in-script
#       sys.path.append(os.path.dirname(os.path.abspath(__file__))) adds the wrong dir.
ENV PYTHONPATH="/app:/app/data_pipeline:/app/data_pipeline/blender_scripts"

# NVIDIA runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENTRYPOINT ["bash", "/app/data_pipeline/k8s/entrypoint.sh"]
