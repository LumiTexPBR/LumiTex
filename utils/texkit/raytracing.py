"""Ray tracing abstraction with pluggable backends (optix, aprmis, nvdiffrast)."""
import os
from typing import Tuple
import torch


if len(os.environ.get('OptiX_INSTALL_DIR', '')) > 0:
    DEFAULT_RAY_TRACING_BACKEND = 'optix'
else:
    DEFAULT_RAY_TRACING_BACKEND = 'aprmis'


class RayTracing:
    """Unified ray-mesh intersection with selectable backend.

    Parameters
    ----------
    vertices : torch.Tensor
        Vertex positions, shape ``[V, 3]``, float32.
    faces : torch.Tensor
        Triangle indices, shape ``[F, 3]``, int64.
    backend : str
        One of ``'optix'``, ``'aprmis'``/``'slang'``, ``'nvdiffrast'``/``'nvdiff'``.
    """

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor,
                 backend: str = DEFAULT_RAY_TRACING_BACKEND, **kwargs):
        V, _ = vertices.shape
        F, _ = faces.shape

        if backend in ('optix', 'triro'):
            from triro.ray.ray_optix import RayMeshIntersector
            self.ray_tracing = RayMeshIntersector(vertices=vertices, faces=faces)
        elif backend in ('aprmis', 'slang'):
            from utils.texkit._vendor.raytracing.rt_aprmis import APRMISRayTracing
            self.ray_tracing = APRMISRayTracing(vertices=vertices, faces=faces)
        elif backend in ('nvdiffrast', 'nvdiff'):
            from utils.texkit._vendor.raytracing.rt_nvdiffrast import NVDiffrastRayTracing
            self.ray_tracing = NVDiffrastRayTracing(
                vertices=vertices, faces=faces,
                cuda_or_gl=kwargs.get('cuda_or_gl', True),
                H=kwargs.get('H', 512),
                W=kwargs.get('W', 512),
                perspective=kwargs.get('perspective', True),
                fov=kwargs.get('fov', 49.1),
                near=kwargs.get('near', 0.01),
                far=kwargs.get('far', 1000.0),
            )
        else:
            raise NotImplementedError(f"backend '{backend}' is not supported")

        self.backend = backend
        self.V = V
        self.F = F

    def update_raw(self, vertices: torch.Tensor, faces: torch.Tensor):
        """Replace the mesh geometry for subsequent queries."""
        return self.ray_tracing.update_raw(vertices=vertices, faces=faces)

    def intersects_closest(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find closest ray-triangle intersections.

        Returns
        -------
        hit, front, tri_idx, loc, uv
        """
        if self.backend in ('optix', 'triro'):
            return self.ray_tracing.intersects_closest(
                origins=rays_o, directions=rays_d, stream_compaction=False)
        else:
            return self.ray_tracing.intersects_closest(rays_o=rays_o, rays_d=rays_d)


def ray_tracing(
    vertices: torch.Tensor, faces: torch.Tensor,
    rays_o: torch.Tensor, rays_d: torch.Tensor,
    backend: str = 'optix',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot ray-mesh intersection (convenience wrapper)."""
    return RayTracing(vertices, faces, backend=backend).intersects_closest(rays_o, rays_d)
