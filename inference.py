import torch
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import os
from typing import Dict, List, Optional
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from diffusers import DiffusionPipeline
import argparse
import json
import logging
from lumitex.transformer.modules import CustomFluxTransformer2DModel
from utils.image_utils import make_image_grid, grid_to_images, extract_channel, adjust_albedo_tone_mapping
from utils.logger import get_logger
from utils.nvdiffrast_utils import *
from utils.texkit.render import VideoExporter, NVDiffRendererInverse
from utils.texkit.image import preprocess
from utils.texkit.uv import preprocess_blank_mesh
from utils.texkit.mesh_io import mesh_uv_wrap
from utils.texkit.convert import tensor_to_image, image_to_tensor
import utils.texkit.raytracing as rt_module
from utils.mesh_utils import remesh_mesh
from Renderer.DifferentiableRenderer.MeshRender import MeshRender
from Renderer.DifferentiableRenderer.mesh_utils import convert_obj_to_glb
from utils.pipeline_utils import ViewProcessor
# from utils.flux_utils import use_compile
from Renderer.renderer_lvsm.render_orth import export_lvsm_condition
from lvsm.inference import load_multiview_data, save_rendered_images, setup_model

import gc

rt_module.DEFAULT_RAY_TRACING_BACKEND = 'nvdiffrast'

logger = get_logger(__name__)

"""
    FluxMVPipeline support two input modes:
        - [RELEASED] mesh: mesh input
        - [DEBUG] cond_imgs: condition images
    
    For mesh mode, please put the mesh in the directory `input_dir` with the structure:
        - {input_dir}/mesh.*: mesh file
        - {input_dir}/ref.png: reference image
        - {input_dir}/tmp/: temporary directory for multiview renderings
    
    For cond_imgs mode, please put the condition images in the directory `input_dir` with the following structure:
        - {input_dir}/normal/{idx:03d}.png: normal images
        - {input_dir}/ccm/{idx:03d}.png: position images
        - {input_dir}/masks/{idx:03d}.png: mask images
        - {input_dir}/ref.png: reference image
"""


# https://github.com/Tencent/Hunyuan3D-2/blob/03cb05a50881cf025b9368aa5f7396cfaaf8ccab/hy3dgen/texgen/hunyuanpaint/pipeline.py#L290
def read_img(img_path: str, bg_c: float=1.0):
    img = imageio.imread(img_path)
    img = img.astype(np.float32) / 255.0
    mask = None
    if img.shape[-1] == 4:
        alpha = img[..., -1:]
        v = alpha
        img = img[..., :-1] * v + bg_c * (1 - v)
        mask = (alpha > 0).astype(np.float32) # (h, w, 1)
    image = Image.fromarray(to8b(img))
    return image, mask

def to8b(img):
    return (img * 255).clip(0, 255).astype(np.uint8)

class RMBG2:
    def __init__(self):
        ckpt_id = "briaai/RMBG-2.0"
        model = AutoModelForImageSegmentation.from_pretrained(ckpt_id, trust_remote_code=True)
        model.to('cuda').eval()
        self.model = model

        # Data settings
        image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_tensor = transforms.ToPILImage()

    def __call__(self, image:Image.Image) -> Image.Image:
        image = image.convert('RGB')
        input_images = self.transform_image(image).unsqueeze(0).to('cuda')
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = self.transform_tensor(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        return image


class FluxMVPipelineBase:
    def __init__(
        self,
        pretrained_model: str = None,
        custom_pipeline: str = "lumitex",
        pbr_mode: bool = False,
        resolution: int = 512,
        input_mode: str = "mesh", # `cond_imgs`, `mesh`
        seed: int = 66,
        logger: logging.Logger = get_logger(__name__),
    ):
        self.pretrained_model = pretrained_model
        self.custom_pipeline = custom_pipeline
        self.resolution = resolution
        self.pbr_mode = pbr_mode
        self.pipe = None
        self.input_mode = input_mode
        self.generator = torch.Generator().manual_seed(seed)
        assert input_mode in ["cond_imgs", "mesh", "debug"], f"Invalid input mode: {input_mode}"
        
        self.logger = logger
        self.device = "cuda"
        self.bg_c = 0.5
        
        # Load FluxMVPipeline modules
        if input_mode != "debug":
            self.load_pipe()
        self.rembg_session = RMBG2()
        # Init renderer
        self.video_exporter = VideoExporter()
        self.inverse_renderer = NVDiffRendererInverse(device=self.device)
        self.render_size = 2048
        self.renderer = MeshRender(
            default_resolution=self.render_size,
            camera_distance=2.8,
            texture_size=2048,
            bake_mode="back_sample",
            shader_type="face",
            raster_mode="cr",
        )
        self.view_processor = ViewProcessor(render=self.renderer)
        self.cam_elevs = [20, 20, 20, 20, -20, -20]
        self.cam_azims = [0, 90, 180, 270, 330, 30]
        self.cam_weights = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05]
    
    def to(self, device: str, dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.pipe = self.pipe.to(device=device, dtype=dtype)
        return self
    
    def load_pipe(self, fuse_qkv_projections: bool = False, use_compile: bool = False):
        self.pipe = DiffusionPipeline.from_pretrained(
            self.pretrained_model,
            custom_pipeline=self.custom_pipeline,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        if self.pbr_mode:
            self.pipe.transformer = CustomFluxTransformer2DModel(
                self.pipe.transformer.transformer,
                pbr_mode=self.pbr_mode,
            )
        # self.pipe = self.pipe.to(self.device, torch.bfloat16)
        if fuse_qkv_projections:
            self.pipe.transformer.fuse_qkv_projections()
            self.pipe.vae.fuse_qkv_projections()
        if use_compile:
            self.pipe = use_compile(self.pipe)

    def load_single_lora_adapter(self, lora_path: str, adapter_name: str, adapter_weights: float = 1.0, force_reinit: bool = False):
        """Load a single LoRA adapter, ensuring complete isolation from other adapters"""
        from safetensors.torch import load_file
        
        if force_reinit:
            if hasattr(self, 'pipe') and self.pipe is not None:
                del self.pipe
        
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            torch.manual_seed(66)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(66)
                torch.cuda.manual_seed_all(66)
            
            self.load_pipe()
            self.pipe = self.pipe.to(device=self.device, dtype=torch.bfloat16)
            self.generator = torch.Generator().manual_seed(66)
        else:
            self.unload_lora_weights()
            gc.collect()
            torch.cuda.empty_cache()
            self.generator = torch.Generator().manual_seed(66)
            
        # Load the new adapter
        # rename_lora_params(lora_path)
        self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        self.pipe.set_adapters([adapter_name], adapter_weights=[adapter_weights])
        # FIXME: load weights include `learned`, 'dino_embedder'
        params = {}
        params_ = load_file(lora_path, device='cpu')
        for k, v in params_.items():
            if 'dino_embedder' in k or 'learned' in k:
                params[k] = adapter_weights * v
        self.pipe.transformer.load_state_dict(params, strict=False)
        self.logger.info(f"Loaded learned weights: {params.keys()}")
        
        self.pipe = self.pipe.to(device=self.device, dtype=torch.bfloat16)
        self.logger.info(f"Loaded single LoRA adapter '{adapter_name}' from {lora_path}")


    def unload_lora_weights(self):
        """[DEPRECATED] Unload all LoRA weights and adapters"""
        try:
            self.pipe.unload_lora_weights()
            active_adapters = self.pipe.get_active_adapters()
            if active_adapters:
                self.pipe.delete_adapters(active_adapters)
            self.logger.info("Successfully unloaded all LoRA weights and adapters.")
        except Exception as e:
            self.logger.warning(f"Failed to unload LoRA weights: {e}")

    def read_img(self, path: str, bg_c = None) -> Image.Image:
        if bg_c is None:
            bg_c = self.bg_c
        img, mask = read_img(path)
        img = np.array(img) / 255.
        if mask is not None:
            mask = np.array(mask) / 255.
            if mask.shape[-1] == 1:
                mask = mask.repeat(3, axis=-1)
            img[mask == 0] = bg_c
        img = Image.fromarray(to8b(img)).resize((self.resolution, self.resolution))
        return img

    def read_imgs(self, path: str, bg_c = None) -> List[Image.Image]:
        imgs = []
        for idx in range(6):
            img = self.read_img(f"{path}/{idx:03d}.png", bg_c)
            imgs.append(img)
        return imgs
    
    def preprocess_reference_image(self, input_image: Image.Image, save_dir: str, scale: float = 0.95, color: str = 'grey'):
        input_image = input_image.convert('RGB').resize((1024, 1024))
        output_image = preprocess(
            input_image,
            alpha=None,
            H=1024,
            W=1024,
            scale=scale,
            color=color,
            return_alpha=False,
            rembg_session=self.rembg_session,
        )
        output_image.save(os.path.join(save_dir, 'rmbg_image.png'))
        output_image = output_image.convert('RGB').resize((1024, 1024))
        output_image.save(os.path.join(save_dir, 'processed_image.png'))
        return output_image
    
    def preprocess_blank_mesh(self, save_dir, input_mesh_path, min_faces=20_000, max_faces=200_000, scale=0.95):
        output_mesh_path = os.path.join(save_dir, 'processed_mesh.obj')
        preprocess_blank_mesh(
            input_mesh_path,
            output_mesh_path,
            min_faces=min_faces,
            max_faces=max_faces,
            scale=scale,
        )
        return output_mesh_path
    
    def render_geometry_images(self, save_dir, input_mesh_path):
        radius = 2.8
        c2ws = generate_orbit_views_c2ws_from_elev_azim(radius=radius, elevation=self.cam_elevs, azimuth=self.cam_azims)
        out = self.video_exporter.export_condition(
            input_mesh_path,
            geometry_scale=0.90,
            n_views=6,
            n_rows=2,
            n_cols=3,
            H=1024,
            W=1024,
            fov_deg=49.1,
            scale=1.0,
            perspective=False,
            orbit=False,
            c2ws=c2ws,
            background='grey',
            return_info=False,
            return_image=True,
            return_mesh=True,
            return_camera=True,
            normal_map_strength=1.0
        )

        out['alpha'].save(os.path.join(save_dir, 'test_mv_alpha.png'))
        out['ccm'].save(os.path.join(save_dir, 'test_mv_ccm.png'))
        out['normal'].save(os.path.join(save_dir, 'test_mv_normal.png'))
        out['normal_bump'].save(os.path.join(save_dir, 'test_mv_normal_bump.png'))
        out['mesh'] = out['mesh'].scale(0.5).apply_transform() # To fit with Hunyuan3D-2.1 Renderer intrinsics
        return out

class FluxMVPipeline(FluxMVPipelineBase):
    def __init__(
        self,
        pretrained_model: str = None,
        custom_pipeline: str = "lumitex",
        pbr_mode: bool = False,
        resolution: int = 512,
        input_mode: str = "cond_imgs", # `cond_imgs`, `mesh`
        seed: int = 66,
        save_suffix: str = "",
        logger: logging.Logger = get_logger(__name__),
    ):
        super().__init__(pretrained_model, custom_pipeline, pbr_mode, resolution, input_mode, seed, logger)
        self.save_suffix = save_suffix if save_suffix is not None else ""
        self.infer_mode_list = ["AL", "MR"]
        self.infer_mode = self.infer_mode_list[0]
        self.lora_path = {}
        

    def back_from_multiview_weighted(self, images: List[Image.Image], cams: Dict, bake_exp: float = 4.0):
        assert len(images) == len(cams['elev']) == len(cams['azim']) == len(cams['weights']), \
            f"Length of images, cams['elev'], cams['azim'], cams['weights'] must be the same."
        project_textures = []
        project_weighted_cos_maps = []
        for view, camera_elev, camera_azim, weight in zip(images, cams['elev'], cams['azim'], cams['weights']):
            project_texture, project_cos_map, project_boundary_map = self.renderer.back_project(
                view.resize((self.render_size, self.render_size)), camera_elev, camera_azim,
            )
            project_cos_map = weight * (project_cos_map**bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            texture, ori_trust_map = self.renderer.fast_bake_texture(project_textures, project_weighted_cos_maps)
        return texture, ori_trust_map > 1e-8
    
    def inference(
        self,
        ref_image: Image.Image,
        guidance_scale: float = 3.5,
        **cond_images: Dict[str, List[Image.Image]],
    ):
        """
            Inference mode: {"RGB": Shaded images, "AL": Albedo, "MR": Metallic, Roughness}
        """
        prompt_embeds = torch.zeros((1, 128, 4096), dtype=torch.bfloat16, device=self.device)
        pooled_prompt_embeds = torch.zeros((1, 768), dtype=torch.bfloat16, device=self.device)
        num_in_batch = len(cond_images["normal_imgs"])
        assert num_in_batch == 6, f"Only support 6 images in one batch."
        # resize
        img_size = (self.resolution, self.resolution)
        cached_condition = {
            "num_in_batch": num_in_batch,
            "normal_imgs": [[im.resize(img_size) for im in cond_images["normal_imgs"]]],
            "position_imgs": [[im.resize(img_size) for im in cond_images["position_imgs"]]],
            "mask_imgs": [[im.resize(img_size) for im in cond_images["mask_imgs"]]],
            "ref_img": [[ref_image.resize(img_size)]],
        }

        self.logger.info(f"Generating images with {num_in_batch} views. Infer mode: {self.infer_mode}.")
        out = self.pipe(
            image=ref_image.resize(img_size),
            height=self.resolution,
            width=self.resolution,
            num_inference_steps=28,
            guidance_scale=guidance_scale,
            generator=self.generator,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            infer_mode=self.infer_mode,
            experimental=False,
            **cached_condition,
        ).images
        return out

    def switch_lora_adapter(self, target_mode: str, force_reinit: bool = False):
        """Switch to a specific LoRA adapter by completely reloading it"""
        if target_mode not in self.infer_mode_list:
            raise ValueError(f"Target adapter '{target_mode}' not in available adapters: {self.infer_mode_list}")
        if target_mode not in self.lora_path:
            raise ValueError(f"No LoRA path configured for mode '{target_mode}'")
        
        # Completely reload the target adapter
        self.load_single_lora_adapter(self.lora_path[target_mode], adapter_name=target_mode, force_reinit=force_reinit)
        self.logger.info(f"Switched to LoRA adapter '{target_mode}'")

    def load_lora_weights(self, lora_path: str, adapter_name: str, adapter_weights: float = 1.0, force_reinit: bool = False):
        """Backward compatibility method - delegates to load_single_lora_adapter"""
        self.load_single_lora_adapter(lora_path, adapter_name, adapter_weights, force_reinit=force_reinit)

    def fuse_lora(self, adapter_names: Optional[List[str]] = None, lora_scale: float = 1.0):
        self.pipe.fuse_lora(
            adapter_names=adapter_names,
            lora_scale=lora_scale,
            components=["transformer"],
        )
        
    def unload_lora_weights(self):
        self.pipe.unload_lora_weights()
        
    def save_pretrained(self, output_dir: str):
        self.pipe.transformer.save_pretrained(output_dir)

    def _call_cond_imgs(self, input_dir: str, output_dir: str):
        ref_image = self.read_img(f"{input_dir}/ref.png")
        normal_imgs = self.read_imgs(f"{input_dir}/normal")
        position_imgs = self.read_imgs(f"{input_dir}/ccm")
        mask_imgs = self.read_imgs(f"{input_dir}/masks", bg_c=0.)

        os.makedirs(output_dir, exist_ok=True)
        make_image_grid(normal_imgs, rows=2, cols=3, resize=512).save(f"{output_dir}/test_normal_{self.save_suffix}.png")
        make_image_grid(position_imgs, rows=2, cols=3, resize=512).save(f"{output_dir}/test_position_{self.save_suffix}.png")
        
        processed_ref_image = self.preprocess_reference_image(ref_image, save_dir=output_dir)
        
        images = self.inference(
            ref_image=processed_ref_image,
            infer_mode=self.infer_mode,
            normal_imgs=normal_imgs,
            position_imgs=position_imgs,
            mask_imgs=mask_imgs,
        )
        
        return images
    
    
    def _call_mesh(self, input_dir: str, output_dir: str, use_atlas: bool = False):
        # TODO: `output_dir` hard code for now
        ref_image = self.read_img(f"{input_dir}/ref.png")
        mesh_files = [f for f in os.listdir(input_dir) if f.endswith(('.obj', '.glb'))]
        
        if not mesh_files:
            raise FileNotFoundError(f"No mesh file found in {input_dir}. Expected files with .obj or .glb extension")
        assert len(mesh_files) == 1, f"Only support one mesh file in {input_dir}. Expected files with .obj or .glb extension"
        
        mv_tmp_dir = os.path.join(output_dir, 'tmp')
        os.makedirs(mv_tmp_dir, exist_ok=True)
        input_mesh_path = os.path.join(input_dir, mesh_files[0])
        # output_mesh_path = self.preprocess_blank_mesh(save_dir=mv_tmp_dir, input_mesh_path=input_mesh_path)
        processed_ref_image = self.preprocess_reference_image(ref_image, save_dir=mv_tmp_dir)
        out = self.render_geometry_images(save_dir=mv_tmp_dir, input_mesh_path=input_mesh_path)
        
        normal_imgs = grid_to_images(out['normal'])
        normal_bump_imgs = grid_to_images(out['normal_bump'])
        position_imgs = grid_to_images(out['ccm'])
        mask_imgs = grid_to_images(out['alpha'])
        mesh = out['mesh']
        cam_info = {'c2ws': out['c2ws'], 'intrinsics': out['intrinsics']}
        mesh.export(output_mesh_path := os.path.join(mv_tmp_dir, 'processed_mesh.obj'))
        
        images = {}
        for mode in self.infer_mode_list:
            self.switch_lora_adapter(mode)
            self.infer_mode = mode
            self.resolution = 512
            guidance_scale = 3.5
            _images = self.inference(
                ref_image=processed_ref_image,
                guidance_scale=guidance_scale,
                normal_imgs=normal_bump_imgs if mode == "AL" and not use_atlas else normal_imgs,
                position_imgs=position_imgs,
                mask_imgs=mask_imgs,
            )
            images[mode] = _images
        
        if out.get('normal_map_texture', None) is not None:
            images['normal_map_texture'] = out['normal_map_texture']
        
        return images, output_mesh_path, mask_imgs
    
    
    def __call__(self, input: str, output_dir: str, inpaint: bool = False, use_atlas: bool = False):
        # multiview generation
        if self.input_mode == "cond_imgs": # DEBUG ONLY
            images = self._call_cond_imgs(input, output_dir)
            make_image_grid(images, rows=2, cols=3, resize=512).save(f"{output_dir}/result_mv.png")
            return
        if self.input_mode == "mesh":
            images, mesh_path, mask_imgs = self._call_mesh(input, output_dir, use_atlas)
        
        for k in self.infer_mode_list:
            v = images[k]
            make_image_grid(v, rows=2, cols=3, resize=512).save(f"{output_dir}/result_mv_{k}.png")
                
        self.logger.success(f"Images saved to {output_dir}.")
        
        # save mesh
        if self.input_mode == "mesh":
            mesh = trimesh.load(mesh_path)
            """
                NOTE: use_atlas & handle bump map
                use_atlas to avoid uv wrap issues, but bump map will be lost
            """
            if use_atlas:
                mesh = mesh_uv_wrap(mesh)
            else:
                if images.get('normal_map_texture', None) is not None:
                    self.renderer.set_texture_normal(images['normal_map_texture'], force_set=True)
                    self.logger.info(f"Set normal map texture to renderer.")
                else:
                    self.logger.warning("No normal map texture found in mesh.")
            self.renderer.load_mesh(mesh=mesh, auto_center=False)
            # Albedo
            bake_cams = {
                'elev': self.cam_elevs[:6],
                'azim': self.cam_azims[:6],
                'weights': self.cam_weights[:6]
            }
            texture, mask = self.back_from_multiview_weighted(
                images["AL"],
                cams=bake_cams,
            )
            texture = self.view_processor.texture_inpaint(texture, mask)
            # FOR PAPER VISUALIZATION
            # texture = self.view_processor.texture_inpaint(texture, mask_np, default=torch.tensor([171/255, 0.0, 171/255])) # blender `magenta`
            self.renderer.set_texture(texture, force_set=True)
            # MR
            if images.get("MR", None) is not None:
                texture_mr, mask_mr = self.back_from_multiview_weighted(images["MR"], cams=bake_cams)
                mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
                texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
                texture_mr = texture_mr[..., [1, 2, 0]]
                self.renderer.set_texture_mr(texture_mr)
            # save mesh
            self.renderer.save_mesh(output_mesh_path := os.path.join(output_dir, 'tmp/textured_mesh.obj'), downsample=False)
            convert_obj_to_glb(output_mesh_path, output_mesh_path.replace('.obj', '.glb'))
            self.logger.success(f"Mesh saved to {output_dir}.")
            
            if inpaint:
                candidate_views_path = "lvsm/config/candidate_views_48_tmp.json"
                inpaint_res = 768
                lvsm_tmp_dir = os.path.join(output_dir, 'tmp/lvsm')
                os.makedirs(lvsm_tmp_dir, exist_ok=True)
                input_mesh_path = os.path.join(output_dir, 'tmp/textured_mesh.obj')
                export_lvsm_condition(cadidate_views_path=candidate_views_path, model_path=input_mesh_path, output_dir=lvsm_tmp_dir, H=1024, W=1024, geometry_scale=0.90)

                has_mr = images.get("MR") is not None
                channels = {"albedo": images["AL"]}
                if has_mr:
                    channels["mr"] = images["MR"]

                for name, imgs in channels.items():
                    channel_dir = os.path.join(lvsm_tmp_dir, name)
                    os.makedirs(channel_dir, exist_ok=True)
                    for i, img in enumerate(imgs):
                        img.resize((inpaint_res, inpaint_res)).save(os.path.join(channel_dir, f"{i:03d}.png"))

                lvsm_model = setup_model("lvsm/config/lact_l24_d768_ttt2x.yaml", inpaint_res, model_dir="ckpt/FLUX-MV-Shaded")
                channel_dirs = {}
                for name in channels:
                    data = load_multiview_data(lvsm_tmp_dir, inpaint_res, name)
                    for k, v in data.items():
                        data[k] = v.unsqueeze(0).cuda()
                    input_data = {k: v[:, :6] for k, v in data.items()}
                    target_data = {k: v[:, 6:] for k, v in data.items()}
                    self.logger.info(f"Running LVSM inference for {name}...")
                    with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True), torch.no_grad():
                        rendering = lvsm_model(input_data, target_data)
                    channel_dir = os.path.join(lvsm_tmp_dir, name)
                    save_rendered_images(rendering.float(), channel_dir)
                    channel_dirs[name] = channel_dir

                cams = json.load(open(candidate_views_path))
                self.cam_elevs = cams['a']
                self.cam_azims = cams['b']
                self.cam_weights = cams['c']
                bake_cams = {
                    'elev': self.cam_elevs,
                    'azim': self.cam_azims,
                    'weights': self.cam_weights,
                }

                lvsm_images = {}
                for name, channel_dir in channel_dirs.items():
                    lvsm_images[name] = [
                        Image.open(os.path.join(channel_dir, f"{idx:03d}.png"))
                        for idx in range(len(self.cam_elevs))
                    ]

                texture_al, mask_al = self.back_from_multiview_weighted(lvsm_images['albedo'], cams=bake_cams)
                mask_np_al = (mask_al.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
                texture_al = self.view_processor.texture_inpaint(texture_al, mask_np_al, mask_gray=None)
                self.renderer.set_texture(texture_al, force_set=True)

                if has_mr:
                    texture_mr, mask_mr = self.back_from_multiview_weighted(lvsm_images['mr'], cams=bake_cams)
                    mask_np_mr = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
                    texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_np_mr)
                    texture_mr = texture_mr[..., [1, 2, 0]]
                    self.renderer.set_texture_mr(texture_mr)

                self.renderer.save_mesh(output_mesh_path := os.path.join(output_dir, 'tmp/lvsm/textured_mesh.obj'), downsample=False)
                convert_obj_to_glb(output_mesh_path, output_mesh_path.replace('.obj', '.glb'))
                self.logger.success(f"Mesh saved to {output_dir}.")

        return images


    def _process_material_channel(self, images, mask_imgs, channel_idx, scale_factor, output_dir, channel_name):
        """Process a single material channel (metallic or roughness)
            Return:
                - channel_images: tensor (B, H, W, 3)
        """
        # Extract channel and convert to tensor
        channel_tensor = image_to_tensor(extract_channel(images, channel_idx))
        mask_imgs = [im.resize((512, 512)) for im in mask_imgs]
        masks = image_to_tensor(mask_imgs)[..., 0:1]
        
        # Apply tone mapping with proper dimension handling
        channel_tensor = channel_tensor.permute(0, 3, 1, 2)
        masks = masks.permute(0, 3, 1, 2)
        
        processed_tensor = adjust_albedo_tone_mapping(
            channel_tensor, masks, scale_factor=scale_factor
        ).permute(0, 2, 3, 1)
        
        # Convert back to images and save
        channel_images = tensor_to_image(processed_tensor)
        make_image_grid(
            channel_images, rows=2, cols=3, resize=512
        ).save(f"{output_dir}/result_mv_{channel_name}.png")
        
        return processed_tensor    
    
    def _debug_renderer(self, mv_img_path: str, mesh_path: str):
        mv_img = Image.open(mv_img_path)
        images = grid_to_images(mv_img, n_rows=2, n_cols=3)
        radius = 2.8
        scale = 1.0
        c2ws = generate_orbit_views_c2ws_from_elev_azim(radius=radius, elevation=self.cam_elevs, azimuth=self.cam_azims)
        intrinsics = generate_intrinsics(scale, scale, fov=False, degree=False)
        cam_info = {'c2ws': c2ws.to(device='cuda'), 'intrinsics': intrinsics.to(device='cuda')}
        debug_dir = os.path.dirname(mv_img_path)
        textured_mesh = self.back_from_multiview_weighted(debug_dir, mesh_path, images, cam_info)
        textured_mesh.export(os.path.join(debug_dir, 'textured_mesh.glb'))
        self.logger.success(f"Mesh saved to {os.path.join(debug_dir, 'textured_mesh.glb')}.")
        
    def _debug_render_condition(self, input_mesh_path: str, output_dir: str):
        # mesh = trimesh.load(input_mesh_path)
        # self.renderer.load_mesh(mesh=mesh, auto_center=False)
        out = self.render_geometry_images(save_dir=output_dir, input_mesh_path=input_mesh_path, backend="unitex")
        self.logger.success(f"Render condition saved to {output_dir}.")

    def _debug_back_from_multiview(self, mv_img_path: str, mesh_path: str, output_dir: str):
        mv_img = Image.open(mv_img_path)
        images = grid_to_images(mv_img, n_rows=2, n_cols=3)
        mesh = trimesh.load(mesh_path)
        mesh = mesh_uv_wrap(mesh)
        self.renderer.load_mesh(mesh=mesh, auto_center=False)
        out = self.render_geometry_images(save_dir=output_dir, input_mesh_path=mesh_path, backend="hunyuan")
        texture, mask = self.back_from_multiview_weighted(images, bake_exp=4.0)
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.renderer.set_texture(texture, force_set=True)
        self.renderer.save_mesh(output_mesh_path := os.path.join(output_dir, 'textured_mesh.obj'), downsample=True)
        convert_obj_to_glb(output_mesh_path, output_mesh_path.replace('.obj', '.glb'))
        self.logger.success(f"Mesh saved to {output_dir}.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folder = args.folder
    
    input_mode = "mesh"
    pbr_mode = "MR"
    save_suffix = None
    
    pipe = FluxMVPipeline(
        pretrained_model="ckpt/FLUX-MV-Shaded", # "Jingzhi/Flux-MV-Shaded",
        custom_pipeline="lumitex",
        pbr_mode=pbr_mode,
        resolution=512,
        input_mode=input_mode,
        save_suffix=save_suffix,
    )
    pipe.infer_mode_list = ["AL"] # Add "MR" for metallic & roughness generation
    pipe.lora_path = {
        "AL": "./ckpt/FLUX-MV-Shaded/pytorch_lora_weights_al.safetensors",
        "MR": "./ckpt/FLUX-MV-Shaded/pytorch_lora_weights_mr.safetensors",
    }
    
    pipe = pipe.to(device="cuda")
    
    images = pipe(
        input=f"{folder}",
        output_dir=f"{folder}",
        inpaint=True,
        use_atlas=False,
    )
