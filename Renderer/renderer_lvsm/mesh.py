#modified from kiuikit (https://github.com/ashawkey/kiuikit/), thanks to kiui

import os
import cv2
import torch
import trimesh
import numpy as np
from packaging import version

import os
import cv2
import torch
import trimesh
import numpy as np
from packaging import version

from kiui.op import safe_normalize, dot
from kiui.typing import *


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

class Mesh:
    """
    A torch-native trimesh class, with support for ``ply/obj/glb`` formats.

    Note:
        This class only supports one mesh with a single texture image (an albedo texture and a metallic-roughness texture).
    """
    def __init__(
        self,
        v: Optional[Tensor] = None,
        f: Optional[Tensor] = None,
        vn: Optional[Tensor] = None,
        fn: Optional[Tensor] = None,
        vt: Optional[Tensor] = None,
        ft: Optional[Tensor] = None,
        vtng: Optional[Tensor] = None,
        ftng: Optional[Tensor] = None,
        vc: Optional[Tensor] = None, # vertex color
        albedo: Optional[Tensor] = None,
        metallicRoughness: Optional[Tensor] = None,
        bump: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """Init a mesh directly using all attributes.

        Args:
            v (Optional[Tensor]): vertices, float [N, 3]. Defaults to None.
            f (Optional[Tensor]): faces, int [M, 3]. Defaults to None.
            vn (Optional[Tensor]): vertex normals, float [N, 3]. Defaults to None.
            fn (Optional[Tensor]): faces for normals, int [M, 3]. Defaults to None.
            vt (Optional[Tensor]): vertex uv coordinates, float [N, 2]. Defaults to None.
            ft (Optional[Tensor]): faces for uvs, int [M, 3]. Defaults to None.
            vc (Optional[Tensor]): vertex colors, float [N, 3]. Defaults to None.
            albedo (Optional[Tensor]): albedo texture, float [H, W, 3], RGB format. Defaults to None.
            metallicRoughness (Optional[Tensor]): metallic-roughness texture, float [H, W, 3], metallic(Blue) = metallicRoughness[..., 2], roughness(Green) = metallicRoughness[..., 1]. Defaults to None.
            device (Optional[torch.device]): torch device. Defaults to None.
        """
        self.device = device
        self.v = v
        self.vn = vn
        self.vt = vt
        self.f = f
        self.vtng = vtng
        self.ftng = ftng
        self.fn = fn
        self.ft = ft
        # will first see if there is vertex color to use
        self.vc = vc
        # only support a single albedo image
        self.albedo = albedo
        # pbr extension, metallic(Blue) = metallicRoughness[..., 2], roughness(Green) = metallicRoughness[..., 1]
        # ref: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
        self.metallicRoughness = metallicRoughness
        self.bump = bump
        self.ori_center = 0
        self.ori_scale = 1
    
    def __repr__(self):
        out = f'<kiui.mesh.Mesh>'
        if self.v is not None: out += f' v={self.v.shape}'
        if self.f is not None: out += f' f={self.f.shape}'
        if self.vc is not None: out += f' vc={self.vc.shape}'
        if self.albedo is not None: out += f' albedo={self.albedo.shape}'
        if self.metallicRoughness is not None: out += f' metallicRoughness={self.metallicRoughness.shape}'
        if self.bump is not None: out += f' bump={self.bump.shape}'
        return out

    @classmethod
    def load(cls, path, resize=True, clean=False, renormal=False, retex=False, vmap=True, bound=0.9, front_dir='-x', **kwargs):
        """load mesh from path.

        Args:
            path (str): path to mesh file, supports ply, obj, glb.
            clean (bool, optional): perform mesh cleaning at load (e.g., merge close vertices). Defaults to False.
            resize (bool, optional): auto resize the mesh using ``bound`` into [-bound, bound]^3. Defaults to True.
            renormal (bool, optional): re-calc the vertex normals. Defaults to True.
            retex (bool, optional): re-calc the uv coordinates, will overwrite the existing uv coordinates. Defaults to False.
            vmap (bool, optional):  remap vertices based on uv coordinates, so each v correspond to a unique vt. Defaults to True. 
            wotex (bool, optional): do not try to load any texture. Defaults to False.
            bound (float, optional): bound to resize. Defaults to 0.9.
            front_dir (str, optional): front-view direction of the mesh, should be [+-][xyz][ 123]. Defaults to '+z'.
            device (torch.device, optional): torch device. Defaults to None.
        
        Note:
            a ``device`` keyword argument can be provided to specify the torch device. 
            If it's not provided, we will try to use ``'cuda'`` as the device if it's available.

        Returns:
            Mesh: the loaded Mesh object.
        """
        mesh = cls.load_trimesh(path, **kwargs)
    
        # auto-normalize
        if resize:
            mesh.auto_size(bound=bound)
        # print(f"[INFO] load mesh, v: {mesh.v.shape}, f: {mesh.f.shape}")
        
        # auto-fix normal
        if renormal or mesh.vn is None:
            mesh.auto_normal()

        if mesh.vt is None or mesh.ft is None:
            mesh.auto_uv(cache_path=path, vmap=vmap)
        
        mesh.compute_tangents()
        # print(f"[INFO] load mesh, vn: {mesh.vn.shape}, fn: {mesh.fn.shape}")

        # if renormal or mesh.vn is None:
        # print(f"[INFO] load mesh, vn: {mesh.vn.shape}, fn: {mesh.fn.shape}")

        # auto-fix texcoords 
        if retex:
            mesh.auto_uv(cache_path=path, vmap=vmap) 
        
        # if mesh.vt is not None:
            # print(f"[INFO] load mesh, vt: {mesh.vt.shape}, ft: {mesh.ft.shape}")

        # rotate front dir to +z
        if front_dir != "+z":
            # axis switch
            if "-z" in front_dir:
                T = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], device=mesh.device, dtype=torch.float32)
            elif "+x" in front_dir:
                T = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=mesh.device, dtype=torch.float32)
            elif "-x" in front_dir:
                T = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=mesh.device, dtype=torch.float32)
            elif "+y" in front_dir:
                T = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], device=mesh.device, dtype=torch.float32)
            elif "-y" in front_dir:
                T = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=mesh.device, dtype=torch.float32)
            else:
                T = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32)
            # rotation (how many 90 degrees)
            if '1' in front_dir:
                T @= torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32) 
            elif '2' in front_dir:
                T @= torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32) 
            elif '3' in front_dir:
                T @= torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32) 
            mesh.v @= T
            mesh.vn @= T

        return mesh

    @classmethod
    def load_trimesh(cls, path, wotex=False, device=None, process=False):
        """load a mesh using ``trimesh.load()``.

        Can load various formats like ``glb`` and serves as a fallback.

        Note:
            We will try to merge all meshes if the glb contains more than one, 
            but **this may cause the texture to lose**, since we only support one texture image!

        Args:
            path (str): path to the mesh file.
            wotex (bool, optional): do not try to load any texture. Defaults to False.
            device (torch.device, optional): torch device. Defaults to None.

        Returns:
            Mesh: the loaded Mesh object.
        """
        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # use trimesh to load ply/glb
        _data = trimesh.load(path, process=process)
        # always convert scene to mesh, and apply all transforms...
        if isinstance(_data, trimesh.Scene):
            # print(f"[INFO] load trimesh: concatenating {len(_data.geometry)} meshes.")
            # trimesh has built-in function for this
            # _mesh = _data.to_mesh()
            # loop the scene graph and apply transform to each mesh
            _concat = []
            scene_graph = _data.graph.to_flattened() # dict {name: {transform: 4x4 mat, geometry: str}}
            for k, v in scene_graph.items():
                name = v['geometry']
                if name in _data.geometry and isinstance(_data.geometry[name], trimesh.Trimesh):
                    transform = v['transform']
                    _concat.append(_data.geometry[name].apply_transform(transform))
            _mesh = trimesh.util.concatenate(_concat)
        else:
            _mesh = _data
        
        if not wotex:
            if _mesh.visual.kind == 'vertex':
                vertex_colors = _mesh.visual.vertex_colors
                vertex_colors = np.array(vertex_colors[..., :3]).astype(np.float32) / 255
                mesh.vc = torch.tensor(vertex_colors, dtype=torch.float32, device=device)
                # print(f"[INFO] load trimesh: use vertex color: {mesh.vc.shape}")
            elif _mesh.visual.kind == 'texture':
                try:
                    _material = _mesh.visual.material
                    if isinstance(_material, trimesh.visual.material.PBRMaterial):
                        texture = np.array(_material.baseColorTexture).astype(np.float32) / 255
                        # load metallicRoughness if present
                        if _material.metallicRoughnessTexture is not None:
                            metallicRoughness = np.array(_material.metallicRoughnessTexture).astype(np.float32) / 255
                            # NOTE: fix a bug in trimesh that loads metallicRoughness in wrong channels: https://github.com/mikedh/trimesh/issues/2195
                            if version.parse(trimesh.__version__) < version.parse('4.2.2'):
                                metallicRoughness = metallicRoughness[..., [2, 1, 0]]
                            mesh.metallicRoughness = torch.tensor(metallicRoughness, dtype=torch.float32, device=device).contiguous()

                        if _material.normalTexture is not None:
                           bump = np.array(_material.normalTexture).astype(np.float32) / 255 
                           mesh.bump = torch.tensor(bump, dtype=torch.float32, device=device).contiguous()




                    elif isinstance(_material, trimesh.visual.material.SimpleMaterial):
                        texture = np.array(_material.to_pbr().baseColorTexture).astype(np.float32) / 255
                    else:
                        raise NotImplementedError(f"material type {type(_material)} not supported!")
                    if len(texture.shape) == 2:
                        texture = texture[..., None].repeat(3, axis=-1)
                    mesh.albedo = torch.tensor(texture[..., :3], dtype=torch.float32, device=device).contiguous()
                    # print(f"[INFO] load trimesh: load texture: {texture.shape}")
                # there really can be lots of mysterious errors...
                except Exception as e:
                    mesh.albedo = None
                    # print(f"[INFO] load trimesh: failed to load texture.")
            else:
                mesh.albedo = None
                # print(f"[INFO] load trimesh: failed to load texture.")

        vertices = _mesh.vertices

        try:
            texcoords = _mesh.visual.uv
            texcoords[:, 1] = 1 - texcoords[:, 1]
        except Exception as e:
            texcoords = None

        try:
            normals = _mesh.vertex_normals
        except Exception as e:
            normals = None

        # trimesh only support vertex uv...
        faces = tfaces = nfaces = _mesh.faces

        mesh.v = torch.tensor(vertices, dtype=torch.float32).to(device)
        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32).to(device)
            if texcoords is not None
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32).to(device)
            if normals is not None
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32).to(device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32).to(device)
            if texcoords is not None
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32).to(device)
            if normals is not None
            else None
        )

        return mesh


    # aabb
    def aabb(self):
        """get the axis-aligned bounding box of the mesh.

        Returns:
            Tuple[torch.Tensor]: the min xyz and max xyz of the mesh.
        """
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values
    

    @torch.no_grad()
    def auto_size(self, bound=0.9, mode: Literal['box', 'sphere'] = 'box'):
        """auto resize the mesh.

        Args:
            bound (float, optional): resizing into ``[-bound, bound]^3``. Defaults to 0.9.
            mode (Literal['box', 'sphere'], optional): the mode to auto resize the mesh. Defaults to 'box'.
        """
        vmin, vmax = self.aabb()
        self.ori_center = (vmax + vmin) / 2
        if mode == 'box':
            self.ori_scale = 2 * bound / torch.max(vmax - vmin).item()
        elif mode == 'sphere':
            radius = torch.max(torch.norm(self.v - self.ori_center, dim=-1)).item()
            self.ori_scale = bound / radius
        self.v = (self.v - self.ori_center) * self.ori_scale

    
    @torch.no_grad()
    def auto_normal(self):
        """auto calculate the vertex normals.
        """
        i0, i1, i2 = self.f[:, 0].long(), self.f[:, 1].long(), self.f[:, 2].long()
        v0, v1, v2 = self.v[i0, :], self.v[i1, :], self.v[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

        # Splat face normals to vertices
        vn = torch.zeros_like(self.v)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        vn = torch.where(
            dot(vn, vn) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).to(vn.device),
        )
        vn = safe_normalize(vn)

        self.vn = vn
        self.fn = self.f

    @torch.no_grad()
    def auto_uv(self, cache_path=None, vmap=False):
        """auto calculate the uv coordinates.

        Args:
            cache_path (str, optional): path to save/load the uv cache as a npz file, this can avoid calculating uv every time when loading the same mesh, which is time-consuming. Defaults to None.
            vmap (bool, optional): remap vertices based on uv coordinates, so each v correspond to a unique vt (necessary for formats like gltf). 
                Usually this will duplicate the vertices on the edge of uv atlas. Defaults to True.
        """
        # try to load cache
        if cache_path is not None:
            cache_path = os.path.splitext(cache_path)[0] + "_uv.npz"
        if cache_path is not None and os.path.exists(cache_path):
            data = np.load(cache_path)
            vt_np, ft_np, vmapping = data["vt"], data["ft"], data["vmapping"]
        else:
            import xatlas

            v_np = self.v.detach().cpu().numpy()
            f_np = self.f.detach().int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            # chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # save to cache
            if cache_path is not None:
                np.savez(cache_path, vt=vt_np, ft=ft_np, vmapping=vmapping)
        
        vt = torch.from_numpy(vt_np.astype(np.float32)).to(self.device)
        ft = torch.from_numpy(ft_np.astype(np.int32)).to(self.device)
        self.vt = vt
        self.ft = ft

        if vmap:
            vmapping = torch.from_numpy(vmapping.astype(np.int64)).long().to(self.device)
            self.align_v_to_vt(vmapping)

    def remap_uv(self, v):
        """ remap uv texture (vt) to other surface.

        Args:
            v (torch.Tensor): the target mesh vertices, float [N, 3].
        """

        assert self.vt is not None

        if self.v.shape[0] != self.vt.shape[0]:
            self.align_v_to_vt()

        # find the closest face for each vertex
        import cubvh 
        BVH = cubvh.cuBVH(self.v, self.f)
        dist, face_id, uvw = BVH.unsigned_distance(v, return_uvw=True)

        # get original uv
        faces = self.f[face_id].long()
        vt0 = self.vt[faces[:, 0]]
        vt1 = self.vt[faces[:, 1]]
        vt2 = self.vt[faces[:, 2]]

        # calc new uv
        vt = vt0 * uvw[:, 0:1] + vt1 * uvw[:, 1:2] + vt2 * uvw[:, 2:3]

        return vt

    
    def align_v_to_vt(self, vmapping=None):
        """ remap v/f and vn/fn to vt/ft.

        Args:
            vmapping (np.ndarray, optional): the mapping relationship from f to ft. Defaults to None.
        """
        if vmapping is None:
            ft = self.ft.view(-1).long()
            f = self.f.view(-1).long()
            vmapping = torch.zeros(self.vt.shape[0], dtype=torch.long, device=self.device)
            vmapping[ft] = f # scatter, randomly choose one if index is not unique

        self.v = self.v[vmapping]
        self.f = self.ft
        
        if self.vn is not None:
            self.vn = self.vn[vmapping]
            self.fn = self.ft

    @torch.no_grad()
    def compute_tangents(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0,3):
            pos[i] = self.v[self.f[:, i].long()]
            tex[i] = self.vt[self.ft[:, i].long()]
            vn_idx[i] = self.fn[:, i].long()

        tangents = torch.zeros_like(self.vn)
        tansum   = torch.zeros_like(self.vn)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1  = pos[1] - pos[0]
        pe2  = pos[2] - pos[0]
        
        nom   = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
        denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])
        
        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

        # Update all 3 vertices
        for i in range(0,3):
            idx = vn_idx[i][:, None].repeat(1,3)
            tangents.scatter_add_(0, idx, tang)                # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(0, idx, torch.ones_like(tang)) # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = safe_normalize(tangents)
        tangents = safe_normalize(tangents - dot(tangents, self.vn) * self.vn)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))
        self.ftng= self.fn
        self.vtng = tangents
    

    

    def to(self, device):
        """move all tensor attributes to device.

        Args:
            device (torch.device): target device.

        Returns:
            Mesh: self.
        """
        self.device = device
        for name in ["v", "f", "vn", "fn", "vt", "ft", "albedo", "vc", "metallicRoughness", "bump"]:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self
    



