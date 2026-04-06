# texkit

Texture toolkit for the LumiTex inference pipeline. This module is built upon [UniTEX/TextureTools](https://github.com/YixunLiang/UniTEX/tree/master/TextureTools), fixing some bugs (e.g., camera rotation matrix), adding new features (e.g., bump rendering), and re-structured.

## Modules

| Module | Description |
|---|---|
| `convert.py` | `image_to_tensor`, `tensor_to_image` — PIL Image ↔ torch Tensor conversion |
| `image.py` | `preprocess` — background removal, centering, and cropping |
| `mesh_io.py` | Mesh loading (`load_whole_mesh`), UV wrapping (`mesh_uv_wrap`), PBR material linking (`link_rgb_to_mesh`, `link_pbr_to_mesh`) |
| `uv.py` | `preprocess_blank_mesh` — mesh simplification/subdivision + UV atlas generation |
| `raytracing.py` | `RayTracing` — unified ray-mesh intersection (optix / aprmis / nvdiffrast backends) |
| `render.py` | `VideoExporter` — multi-view condition rendering; `NVDiffRendererInverse` — inverse rendering for texture baking |

## `_vendor/`

Vendored internal dependencies used by `raytracing.py` and `render.py`. Preserves the original package structure so all relative imports work unchanged.

| Directory | Description |
|---|---|
| `camera/` | Camera projection matrices, view generation, rotation utilities |
| `geometry/` | Tensor/array conversion helpers, triangle topology operations, UV atlas |
| `image/` | Gaussian blur, lens blur, image fusion, image outpainting |
| `io/` | Mesh file loading, GLB header parsing, OBJ saving, PBR material linking |
| `mesh/` | Mesh data structures (`Texture`, `PBRMesh`), trimesh utilities |
| `pcd/` | Point cloud structure, KNN search (scipy / cupy / faiss backends) |
| `raytracing/` | Ray tracing backends (APRMIS/slangtorch, nvdiffrast) |
| `render/nvdiffrast/` | NVDiffrast-based renderers (base, PBR, inverse) |
| `texture/` | Mipmap pull-push stitching, PBR material model |
| `utils/` | Color parsing |
| `video/` | Multi-view video/condition export |
