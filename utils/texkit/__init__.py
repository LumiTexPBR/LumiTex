"""
texkit — Minimal texture toolkit for the LumiTex inference pipeline.

Modules
-------
convert    : image_to_tensor, tensor_to_image
image      : preprocess (background removal, centering, cropping)
mesh_io    : load_whole_mesh, mesh_uv_wrap, link_rgb_to_mesh, link_pbr_to_mesh
uv         : preprocess_blank_mesh (UV atlas generation)
render     : VideoExporter, NVDiffRendererInverse (lazy-loaded)
raytracing : RayTracing (pluggable backends: optix / aprmis / nvdiffrast)
"""
