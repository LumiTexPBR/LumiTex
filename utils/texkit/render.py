"""Lazy-loaded rendering classes (VideoExporter, NVDiffRendererInverse).

These classes carry deep dependency chains (nvdiffrast renderers, camera
utilities, mesh structures, ray tracing, image processing, …) and are
loaded on first access to avoid pulling in the full dependency graph at
import time.
"""


def __getattr__(name: str):
    if name == "VideoExporter":
        from utils.texkit._vendor.video.export_nvdiffrast_video import VideoExporter
        return VideoExporter
    if name == "NVDiffRendererInverse":
        from utils.texkit._vendor.render.nvdiffrast.renderer_inverse import NVDiffRendererInverse
        return NVDiffRendererInverse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VideoExporter", "NVDiffRendererInverse"]
