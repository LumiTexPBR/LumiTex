# For a consistent evaluation, the Renderer code is based on Hunyuan3D-2.1, as each baseline method may process the mesh differently.
# You may refer to the package path in the following link for more details:
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/tree/main/hy3dpaint/DifferentiableRenderer
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial

from DifferentiableRenderer.MeshRender import MeshRender
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb

try:
    from eval_utils import load_whole_mesh, save_uv_wireframe_image
    from renderer_utils import ViewProcessor
except ImportError:
    from evaluation.eval_utils import load_whole_mesh, save_uv_wireframe_image
    from evaluation.renderer_utils import ViewProcessor


class UVMapEvaluator:
    """
    Evaluate texture quality with a multi-view render and texture-bake pipeline.

    The pipeline renders a method mesh from predefined camera views, bakes those views onto
    a standard mesh, and writes texture/mesh artifacts for inspection.
    """

    def __init__(
        self,
        base_testset_dir: str,
        output_dir: str = None,
        device: str = None,
        seed: int = 42,
    ):
        """
        Initialize evaluator paths, device, renderer, and camera sampling settings.

        If `device` is not provided, CUDA is used when available.
        """
        self.base_testset_dir = base_testset_dir
        self.output_dir = output_dir or os.path.join(base_testset_dir, "uv_evaluation")
        os.makedirs(self.output_dir, exist_ok=True)

        self.seed = seed
        np.random.seed(seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.render_size = 2048 # fixed evaluation resolution
        self.renderer = MeshRender(
            default_resolution=self.render_size,
            camera_distance=2.8,
            texture_size=2048,
            bake_mode="back_sample",
            shader_type="face",
            raster_mode="cr",
        )
        self.view_processor = ViewProcessor(render=self.renderer)

        self.cam_elevs = [0, 0, 0, 0, 90, -90] + np.random.uniform(0.0, 360.0, size=6).tolist()
        self.cam_azims = [0, 90, 180, 270, 0, 0] + np.random.uniform(0.0, 360.0, size=6).tolist()
        self.cam_weights = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

        print("UV Map Evaluator initialized with:")
        print(f"- Render size: {self.render_size}")
        print(f"- Camera views: {len(self.cam_elevs)}")
        print(f"- Device: {self.device}")

    def _set_renderer_texture(self, mesh: trimesh.Trimesh) -> None:
        """
        Set renderer texture from mesh material when available.

        Falls back to renderer default color when no texture is found.
        """
        texture_found = False
        if hasattr(mesh, "visual") and hasattr(mesh.visual, "material"):
            if hasattr(mesh.visual.material, "image") and mesh.visual.material.image is not None:
                self.renderer.set_texture(mesh.visual.material.image)
                texture_found = True
            elif (
                hasattr(mesh.visual.material, "baseColorTexture")
                and mesh.visual.material.baseColorTexture is not None
            ):
                self.renderer.set_texture(mesh.visual.material.baseColorTexture)
                texture_found = True

        if not texture_found:
            print("No texture found in mesh, rendering with default color")

    def render_multiview_images(self, mesh_path: str, resolution: int = 512) -> List[Image.Image]:
        """
        Render one mesh from configured camera elevations/azimuths.

        Returns a list of PIL images in the same order as camera settings.
        """
        print(f"Rendering multiview images for {mesh_path}")

        mesh = load_whole_mesh(mesh_path)
        self.renderer.load_mesh(mesh=mesh, auto_center=True)

        if hasattr(self.renderer, "tex"):
            delattr(self.renderer, "tex")
        self._set_renderer_texture(mesh)

        self.renderer.set_default_render_resolution(resolution)

        multiview_images: List[Image.Image] = []
        for elev, azim in zip(self.cam_elevs, self.cam_azims):
            rendered_image = self.renderer.render_albedo(elev, azim)
            if isinstance(rendered_image, torch.Tensor):
                rendered_image = rendered_image.cpu().numpy()
            if rendered_image.max() <= 1.0:
                rendered_image = (rendered_image * 255).astype(np.uint8)
            multiview_images.append(Image.fromarray(rendered_image))

        self.renderer.set_default_render_resolution(self.render_size)
        print(f"Generated {len(multiview_images)} multiview images")
        return multiview_images

    def bake_texture_from_multiview_images(
        self,
        images: Sequence[Image.Image],
        target_mesh: trimesh.Trimesh,
        bake_exp: float = 4.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Back-project multi-view images to target mesh texture space.

        Each view contributes with weighted cosine confidence before final texture fusion.
        """
        print("Baking texture from multiview images...")

        if hasattr(self.renderer, "tex"):
            delattr(self.renderer, "tex")
        self.renderer.load_mesh(mesh=target_mesh, auto_center=True)

        projected_textures = []
        projected_weighted_cos_maps = []

        for view, camera_elev, camera_azim, weight in zip(
            images, self.cam_elevs, self.cam_azims, self.cam_weights
        ):
            project_texture, project_cos_map, _ = self.renderer.back_project(
                view.resize((self.render_size, self.render_size)), camera_elev, camera_azim
            )
            project_cos_map = weight * (project_cos_map ** bake_exp)
            projected_textures.append(project_texture)
            projected_weighted_cos_maps.append(project_cos_map)

        texture, trust_map = self.renderer.fast_bake_texture(
            projected_textures, projected_weighted_cos_maps
        )
        valid_mask = trust_map > 1e-8

        print("Texture baking completed")
        return texture, valid_mask

    def _inpaint_texture(
        self, baked_texture: torch.Tensor, baked_mask: torch.Tensor
    ) -> np.ndarray:
        """
        Inpaint baked texture with a binary valid-mask and return uint8 RGB texture.
        """
        mask = baked_mask
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        mask_np = (mask.detach().cpu().numpy() > 0).astype(np.uint8) * 255
        processed_texture = self.view_processor.texture_inpaint(
            baked_texture, mask_np, default=0.5
        )
        if isinstance(processed_texture, torch.Tensor):
            processed_texture = processed_texture.detach().cpu().numpy()
        if processed_texture.max() <= 1.0:
            processed_texture = (processed_texture * 255).astype(np.uint8)
        return processed_texture

    def _build_textured_mesh(
        self, standard_mesh: trimesh.Trimesh, texture_image: Image.Image
    ) -> trimesh.Trimesh:
        """
        Attach a texture image to the standard mesh and return a textured mesh copy.
        """
        final_material = SimpleMaterial(image=texture_image)
        textured_mesh = standard_mesh.copy()
        if hasattr(standard_mesh.visual, "uv") and standard_mesh.visual.uv is not None:
            textured_mesh.visual = TextureVisuals(
                uv=standard_mesh.visual.uv, material=final_material
            )
        else:
            textured_mesh.visual = TextureVisuals(material=final_material)
        return textured_mesh

    def _save_outputs(
        self,
        textured_mesh: trimesh.Trimesh,
        texture_image: Image.Image,
        output_case_dir: str,
    ) -> Dict[str, str]:
        """
        Save baked texture, textured mesh, and UV layout visualization artifacts.
        """
        baked_texture_path = os.path.join(output_case_dir, "multiview_baked_texture.png")
        final_obj_path = os.path.join(output_case_dir, "multiview_textured_mesh.obj")
        final_glb_path = final_obj_path.replace(".obj", ".glb")
        uv_layout_path = os.path.join(output_case_dir, "multiview_uv_layout.png")

        texture_image.save(baked_texture_path)
        textured_mesh.export(final_obj_path)
        convert_obj_to_glb(final_obj_path, final_glb_path)
        save_uv_wireframe_image(textured_mesh, uv_layout_path)

        return {
            "baked_texture_path": baked_texture_path,
            "textured_mesh_path": final_glb_path,
            "uv_layout_path": uv_layout_path,
        }

    def evaluate_multiview_texture_quality(
        self,
        case_id: str,
        method: str,
        mesh_filename: str = "tmp/lvsm/textured_mesh.glb",
        output_size: int = 1024,
    ) -> Dict[str, Any]:
        """
        Run the full multi-view evaluation flow for one case and one method.

        Steps: render source views -> bake to standard mesh -> postprocess texture -> export artifacts.
        """
        print(f"\n=== Evaluating {case_id}/{method} with multiview workflow ===")

        method_mesh_path = os.path.join(self.base_testset_dir, case_id, method, mesh_filename)
        standard_mesh_path = os.path.join(self.base_testset_dir, case_id, "mesh", "merged.glb")
        output_case_dir = os.path.join(self.output_dir, case_id, method)
        os.makedirs(output_case_dir, exist_ok=True)

        if not os.path.exists(method_mesh_path):
            return {"success": False, "error": f"method_mesh_not_found: {method_mesh_path}"}
        if not os.path.exists(standard_mesh_path):
            return {"success": False, "error": f"standard_mesh_not_found: {standard_mesh_path}"}

        try:
            print("Loading standard mesh...")
            standard_mesh = load_whole_mesh(standard_mesh_path)

            print("Rendering method mesh views...")
            multiview_images = self.render_multiview_images(method_mesh_path, resolution=512)

            print("Baking rendered views to standard mesh...")
            baked_texture, baked_mask = self.bake_texture_from_multiview_images(
                multiview_images, standard_mesh, bake_exp=4.0
            )
            processed_texture_np = self._inpaint_texture(baked_texture, baked_mask)

            baked_texture_image = Image.fromarray(processed_texture_np).resize(
                (output_size, output_size), Image.LANCZOS
            )
            textured_mesh = self._build_textured_mesh(standard_mesh, baked_texture_image)
            saved_outputs = self._save_outputs(textured_mesh, baked_texture_image, output_case_dir)

            return {
                "success": True,
                "case_id": case_id,
                "method": method,
                "method_mesh_path": method_mesh_path,
                "standard_mesh_path": standard_mesh_path,
                **saved_outputs,
            }
        except Exception as exc:
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(exc)}


def main() -> None:
    """Run a single-case multi-view texture quality evaluation."""
    
    # As baseline methods may process the mesh differently, we take a standard mesh for evaluation.
    # The standard mesh can be taken from handcraft or generated by any method (e.g., Hunyuan3D).
    # Note that the mesh should be merged into a single part (merged.glb) if there are multiple parts.
    # 
    # The evaluation process is as follows:
    # 1. Render the method mesh from predefined camera views.
    # 2. Bake the rendered views to the standard mesh.
    # 3. Postprocess the baked texture.
    # 4. Export the textured mesh and UV layout. (for quantitative evaluation)
    # 
    # Please format your testset directory as follows:
    # testset/
    #   case1/
    #     method1/
    #       textured.glb
    #     method2/
    #       textured.glb
    #     method3/
    #       textured.glb
    #     ...
    #     mesh/
    #       merged.glb
    #     ...
    
    testset_dir = "testset"
    method = "ours_250707"
    case_id = "case1"

    evaluator = UVMapEvaluator(testset_dir, device="cuda", seed=42)
    result = evaluator.evaluate_multiview_texture_quality(
        case_id=case_id,
        method=method,
        mesh_filename="textured.glb",
    )

    if result["success"]:
        print("\n=== Evaluation Completed ===")
        print(f"Case: {case_id}")
        print(f"Baked texture: {result['baked_texture_path']}")
        print(f"Textured mesh: {result['textured_mesh_path']}")
        print(f"UV layout: {result['uv_layout_path']}")
    else:
        print("\n=== Evaluation Failed ===")
        print(f"Case: {case_id}")
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
