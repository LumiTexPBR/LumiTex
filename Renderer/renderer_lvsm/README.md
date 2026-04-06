# renderer_lvsm

GPU-accelerated multi-view renderer for LVSM. Renders 3D meshes (GLB/OBJ) from multiple viewpoints using orthographic projection, producing normal maps, coordinate maps (CCM), and camera parameters as conditioning inputs for downstream models (e.g., LVSM, Flux diffusion).

## Directory Structure

```
renderer_lvsm/
├── render_orth.py          # Main entry point: orthographic multi-view rendering
├── cam.py                  # Camera: ortho/perspective projection, spherical coordinates
├── mesh.py                 # Mesh loading (GLB/OBJ/PLY), auto-normalization, UV generation
├── optix_render/           # NVIDIA OptiX ray-tracing pipeline
│   ├── render.py           #   PBR shading with environment lighting & shadow
│   ├── light.py            #   Environment light: PDF/CDF importance sampling
│   ├── util.py             #   Math utils, color space conversion, pixel grids
│   ├── optixutils/         #   OptiX C++/CUDA wrapper (denoising, env sampling)
│   └── renderutils/        #   BSDF (GGX), loss functions, CUDA kernels
└── pbr/                    # Physically-Based Rendering (cubemap-based)
    ├── shade.py            #   Split-sum PBR shading (diffuse + specular)
    ├── light.py            #   CubemapLight with mipmap specular lobes
    ├── brdf_256_256.bin    #   Pre-computed BRDF LUT (256x256, float32)
    └── renderutils/        #   BSDF implementations, loss, mesh CUDA kernels
```

## Core Components

### `render_orth.py` — Multi-view Orthographic Renderer

Entry point function: `export_lvsm_condition()`

Rasterizes a GLB mesh from N viewpoints (defined by azimuth/elevation angles) using NVDiffRast, and outputs per-view normal maps, position maps (CCM), and OpenCV-format camera parameters.

### `optix_render/` — OptiX Ray-Tracing Pipeline

Full GPU ray-traced rendering with:
- **Environment lighting** with importance sampling (64x64 stratified)
- **PBR shading**: GGX BSDF with shadow rays
- **Depth peeling** for multi-layer rendering
- **Denoising** support via CUDA kernels
- Backward pass gradients for optimization

### `pbr/` — Cubemap-Based PBR Shading

`pbr_shading()` implements the split-sum approximation:
- **Diffuse**: cubemap diffuse lobe lookup * albedo * (1 - metallic)
- **Specular**: roughness-based mipmap specular lobe * Fresnel reflectance (from BRDF LUT)
- **Tone mapping**: ACES film curve + sRGB gamma correction

## Usage

### Python API

```python
from renderer_lvsm.render_orth import export_lvsm_condition

export_lvsm_condition(
    cadidate_views_path="views.json",  # JSON with azimuth/elevation angles
    model_path="mesh.glb",             # GLB model with PBR textures
    output_dir="output/",              # Output directory
    geometry_scale=0.90,               # Mesh normalization bound
    H=512, W=512,                      # Rendering resolution
)
```

### CLI

```bash
python -m renderer_lvsm.render_orth \
    --folder <model_folder> \
    --res 512 \
    --cadidate_views_path views.json
```

### Input: Camera Views JSON

```json
{
    "b": [0.0, 90.0, 180.0, 270.0],
    "a": [20.0, 20.0, 20.0, 20.0]
}
```

- `"b"`: azimuths in degrees
- `"a"`: elevations in degrees

### Output Structure

```
output_dir/
├── normal/
│   ├── 000.png    # Normal map (RGB = normalized XYZ, A = mask)
│   ├── 001.png
│   └── ...
├── ccm/
│   ├── 000.png    # Coordinate/position map (RGB = XYZ, A = mask)
│   ├── 001.png
│   └── ...
└── opencv_cameras.json
```

#### `opencv_cameras.json` Schema

```json
[
    {
        "w": 512, "h": 512,
        "fx": 2.0, "fy": 2.0,
        "cx": 2.0, "cy": 2.0,
        "w2c": [[...], [...], [...], [...]],
        "file_path": "render_type/000.png"
    }
]
```
