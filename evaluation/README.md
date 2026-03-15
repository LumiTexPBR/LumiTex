# Texture Evaluation Notes

This folder contains the evaluation scripts we used for the paper.

The idea is simple:

1. First, map every method output back to the same reference mesh (`stdTexture.py`), so comparison is fair.
2. Then run texture metrics on image files (`run_benchmark.py`).

If we skip step 1, methods with different UV layouts/topology are hard to compare directly.

## 1) What The Pipeline Does

For each case:

1. Use `case{idx}/mesh/merged.glb` as the reference mesh.
2. Render each method mesh from fixed camera views.
3. Bake those views into the reference mesh UV.
4. Inpaint missing texels.
5. Save standardized outputs for inspection and later scoring.

After that, benchmark metrics are computed across cases and methods.

## 2) Expected Data Layout

```text
testset/
  case1/
    mesh/             # standard mesh
      merged.glb
      material_0.png  # GT texture
    methodA/
      textured.glb
      material_0.png
    methodB/
      textured.glb
      material_0.png
  case2/
    ...
```

Standardized outputs from `stdTexture.py` are written to:

```text
testset/uv_evaluation/<case>/<method>/
  multiview_baked_texture.png
  multiview_textured_mesh.obj
  multiview_textured_mesh.glb
  multiview_uv_layout.png
```

## 3) Extra Packages

In addition to the base environment (PyTorch, torchvision, trimesh, Pillow, numpy), install:

```bash
pip install lpips clean-fid
pip install git+https://github.com/openai/CLIP.git
pip install cmmd
```

## 4) How To Run

Step 1: standardize textures onto the reference mesh.

```bash
python evaluation/stdTexture.py
```

Step 2: run metric scoring.

```bash
python evaluation/run_benchmark.py
```

See the `main()` function in each file.
