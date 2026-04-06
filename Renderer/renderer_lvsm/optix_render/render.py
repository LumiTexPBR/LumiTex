# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from threading import local
import numpy as np
import torch
import nvdiffrast.torch as dr
import torch.nn.functional as F
from . import util
from . import renderutils as ru
from . import optixutils as ou
from . import light


rnd_seed = 0

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# ==============================================================================================
#  pixel shader
# ==============================================================================================
def shade(
        rast,
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        view_pos,
        lgt,
        mesh,
        optix_ctx,
        bsdf,
        denoiser,
        shadow_scale
    ):

    # offset = torch.normal(mean=0, std=0.005, size=(gb_depth.shape[0], gb_depth.shape[1], gb_depth.shape[2], 2), device="cuda")
    # jitter = (util.pixel_grid(gb_depth.shape[2], gb_depth.shape[1])[None, ...] + offset).contiguous()

    # mask = (rast[..., -1:] > 0).float()
    # mask_tap = dr.texture(mask.contiguous(), jitter, filter_mode='linear', boundary_mode='clamp')
    # grad_weight = mask * mask_tap

    # ################################################################################
    # # Texture lookups
    # ################################################################################
    # perturbed_nrm = None
    # if 'kd_ks' in material:
    #     # Combined texture, used for MLPs because lookups are expensive
    #     all_tex_jitter = material['kd_ks'].sample(gb_pos + torch.normal(mean=0, std=0.01, size=gb_pos.shape, device="cuda"))
    #     all_tex = material['kd_ks'].sample(gb_pos)
    #     assert all_tex.shape[-1] == 6, "Combined kd_ks must be 6 channels"
    #     kd, ks = all_tex[..., 0:3], all_tex[..., 3:6]
    #     kd_grad  = torch.abs(all_tex_jitter[..., 0:3] - kd)
    #     ks_grad  = torch.abs(all_tex_jitter[..., 3:6] - ks) * torch.tensor([0, 1, 1], dtype=torch.float32, device='cuda')[None, None, None, :] # Omit o-component
    # else:
    #     kd = material['kd'].sample(gb_texc, gb_texc_deriv)
    #     ks = material['ks'].sample(gb_texc, gb_texc_deriv)[..., 0:3] # skip alpha
    #     if 'normal' in material:
    #         perturbed_nrm = material['normal'].sample(gb_texc, gb_texc_deriv)

    #     kd_jitter = dr.texture(kd.contiguous(), jitter, filter_mode='linear', boundary_mode='clamp')
    #     ks_jitter = dr.texture(ks.contiguous(), jitter, filter_mode='linear', boundary_mode='clamp')
    #     kd_grad = torch.abs(kd_jitter - kd) * grad_weight
    #     ks_grad  = torch.abs(ks_jitter - ks) * torch.tensor([0, 1, 1], dtype=torch.float32, device='cuda')[None, None, None, :] * grad_weight # Omit o-component

    # # Separate kd into alpha and color, default alpha = 1
    # alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1]) 
    # kd = kd[..., 0:3]

    alpha = (rast[..., 3:] > 0).float().squeeze(0)
    # alpha = dr.antialias(alpha, rast, v_clip, mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
    alpha = alpha[:,:,0:1]
    # render depth
    # depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f)
    # depth = depth[0]
    # depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)


    # render albedo
    albedo = dr.texture(mesh.albedo.unsqueeze(0), gb_texc, filter_mode='linear') # [1, H, W, 3]
    albedo = albedo[0]

    # render roughness & metallic

    if mesh.metallicRoughness is not None:
        # print(mesh.metallicRoughness.unsqueeze(0).shape)
        metallicRoughness = dr.texture(mesh.metallicRoughness.unsqueeze(0), gb_texc, filter_mode='linear') # [1, H, W, 3]
        metallicRoughness = metallicRoughness[0]
    else:
        metallicRoughness = torch.ones_like(albedo) 
    # metallicRoughness = 0.5*torch.ones_like(albedo)
       

    if mesh.bump is not None:
        bump = dr.texture(mesh.bump.unsqueeze(0), gb_texc, filter_mode='linear')
        # bump = bump[0]

    ################################################################################
    # Normal perturbation & normal bend
    ################################################################################
    # if 'no_perturbed_nrm' in material and material['no_perturbed_nrm']:
    #     perturbed_nrm = None

    # Geometric smoothed normal regularizer
    # nrm_jitter = dr.texture(gb_normal.contiguous(), jitter, filter_mode='linear', boundary_mode='clamp')
    # nrm_grad = torch.abs(nrm_jitter - gb_normal) * grad_weight

    # if perturbed_nrm is not None:
    #     perturbed_nrm_jitter = dr.texture(perturbed_nrm.contiguous(), jitter, filter_mode='linear', boundary_mode='clamp')
    #     perturbed_nrm_grad = 1.0 - util.safe_normalize(util.safe_normalize(perturbed_nrm_jitter) + util.safe_normalize(perturbed_nrm))[..., 2:3]
    #     perturbed_nrm_grad = perturbed_nrm_grad.repeat(1,1,1,3) * grad_weight

    view_pos = gb_pos - view_pos * 5.0
    # view_pos = view_pos.view(1,1,1,3).expand(1,512,512,3)
    # gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, bump, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)

    ################################################################################
    # Evaluate BSDF
    ################################################################################
    

    assert isinstance(lgt, light.EnvironmentLight) and optix_ctx is not None
    ro = gb_pos + gb_normal*0.001

    
    global rnd_seed
    diffuse_accum, specular_accum = ou.optix_env_shade(optix_ctx, rast[..., -1], ro, gb_pos, gb_normal, view_pos, albedo, metallicRoughness, 
                        lgt.base, lgt._pdf, lgt.rows[:,0], lgt.cols, BSDF='pbr-optix', n_samples_x=64, 
                        rnd_seed=None, shadow_scale=shadow_scale)
    rnd_seed += 1

    # denoise demodulated shaded values if possible
    # if denoiser is not None and FLAGS.denoiser_demodulate:
    #     diffuse_accum  = denoiser.forward(torch.cat((diffuse_accum, gb_normal, gb_depth), dim=-1))
    #     specular_accum = denoiser.forward(torch.cat((specular_accum, gb_normal, gb_depth), dim=-1))

    kd = albedo * (1.0 - metallicRoughness[..., 2:3]) # kd * (1.0 - metalness)
    shaded_col = diffuse_accum * kd + specular_accum

    # denoise combined shaded values if possible
    # if denoiser is not None and not FLAGS.denoiser_demodulate:
    #     shaded_col = denoiser.forward(torch.cat((shaded_col, gb_normal, gb_depth), dim=-1))

        
    # Return multiple buffers
    buffers = {
        'shaded'            : torch.cat((shaded_col.squeeze(0), alpha), dim=-1),
        # 'z_grad'            : torch.cat((gb_depth, torch.zeros_like(alpha), alpha), dim=-1),
        'normal'            : torch.cat((gb_normal.squeeze(0), alpha), dim=-1),
        'geometric_normal'  : torch.cat((gb_geometric_normal.squeeze(0), alpha), dim=-1),
        'kd'                : torch.cat((albedo.squeeze(0), alpha), dim=-1),
        'ks'                : torch.cat((metallicRoughness.squeeze(0), alpha), dim=-1),
        # 'kd_grad'           : torch.cat((kd_grad, alpha), dim=-1),
        # 'ks_grad'           : torch.cat((ks_grad, alpha), dim=-1),
        # 'normal_grad'       : torch.cat((nrm_grad, alpha), dim=-1),
        'rast' : rast[..., -1].permute(1,2,0).unsqueeze(0)
    }

    if 'diffuse_accum' in locals():
        buffers['diffuse_light'] = torch.cat((diffuse_accum.squeeze(0), alpha), dim=-1)
    if 'specular_accum' in locals():
        buffers['specular_light'] = torch.cat((5*specular_accum.squeeze(0), alpha), dim=-1)

    if bump is not None: 
        buffers['perturbed_nrm'] = torch.cat((bump.squeeze(0), alpha), dim=-1)
        # buffers['perturbed_nrm_grad'] = torch.cat((perturbed_nrm_grad, alpha), dim=-1)
    return buffers

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        v_pos_clip,
        rast,
        rast_deriv,
        mesh,
        view_pos,
        lgt,
        resolution,
        spp,
        msaa,
        optix_ctx,
        bsdf,
        denoiser,
        shadow_scale
    ):

    full_res = [resolution[0]*spp, resolution[1]*spp]

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out_deriv_s = util.scale_img_nhwc(rast_deriv, resolution, mag='nearest', min='nearest') * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v, rast_out_s, mesh.f)

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)

    v0 = mesh.v[mesh.f[:, 0].long(), :]
    v1 = mesh.v[mesh.f[:, 1].long(), :]
    v2 = mesh.v[mesh.f[:, 2].long(), :]
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())

    # Compute tangent space
    assert mesh.vn is not None and mesh.vtng is not None
    gb_normal, _ = interpolate(mesh.vn[None, ...], rast_out_s, mesh.fn.int())
    gb_tangent, _ = interpolate(mesh.vtng[None, ...], rast_out_s, mesh.ftng.int()) # Interpolate tangents

    # Texture coordinate
    assert mesh.vt is not None
    gb_texc, gb_texc_deriv = interpolate(mesh.vt[None, ...], rast_out_s, mesh.ft.int(), rast_db=rast_out_deriv_s)

    # Interpolate z and z-gradient
    # with torch.no_grad():
    #     eps = 0.00001
    #     clip_pos, clip_pos_deriv = interpolate(v_pos_clip, rast_out_s, mesh.f.int(), rast_db=rast_out_deriv_s)
    #     z0 = torch.clamp(clip_pos[..., 2:3], min=eps) / torch.clamp(clip_pos[..., 3:4], min=eps)
    #     z1 = torch.clamp(clip_pos[..., 2:3] + torch.abs(clip_pos_deriv[..., 2:3]), min=eps) / torch.clamp(clip_pos[..., 3:4] + torch.abs(clip_pos_deriv[..., 3:4]), min=eps)
    #     z_grad = torch.abs(z1 - z0)
    #     gb_depth = torch.cat((z0, z_grad), dim=-1)

    ################################################################################
    # Shade
    ################################################################################

    buffers = shade(rast_out_s, gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_texc,
        view_pos, lgt, mesh, optix_ctx, bsdf, denoiser, shadow_scale)

    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for k in buffers.keys():
            buffers[k] = util.scale_img_nhwc(buffers[k], full_res, mag='nearest', min='nearest')

    # Return buffers
    return buffers


# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
        ctx,
        mesh,
        proj,
        view_pos,
        hdr_path,
        resolution,
        spp        = 1,
        num_layers = 1,
        msaa       = False,
        background = None,
        optix_ctx  = None,
        bsdf       = None,
        denoiser   = None,
        shadow_scale = 1.0
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x

    def composite_buffer(key, layers, background, antialias):
        accum = background
        for buffers, rast, rast_db in reversed(layers):
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
            if antialias:
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.f.int())
        return accum

    assert mesh.f.shape[0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"

    full_res = [resolution[0]*spp, resolution[1]*spp]


    # clip space transform
    # v_pos_clip = ru.xfm_points(mesh.v, mtx_in, use_python=True)
    v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(view_pos).T).float().unsqueeze(0)
    v_pos_clip = v_cam @ proj.T

    # Render all layers front-to-back
    layers = []
    
    R = view_pos[:3, :3]  
    cam_pos = -R[:, 2]  
    # cam_pos = view_pos[3:4,:3]
    cam_pos = cam_pos / torch.norm(cam_pos)
    # cam_pos = cam_pos
    # cam_pos = view_dirs.view(1,1,3).expand(512,512,3)


    lgt = light._load_env_hdr(hdr_path, scale=1, res=[256, 256])

    # Render all layers front-to-back
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.f.int(), full_res) as peeler:
        for _ in range(num_layers):
            rast, rast_db = peeler.rasterize_next_layer()
            layers += [(render_layer(v_pos_clip, rast, rast_db, mesh, cam_pos, lgt, resolution, spp, msaa, optix_ctx, bsdf, denoiser, shadow_scale), rast, rast_db)]

    # Setup background
    if background is not None:
        if spp > 1:
            background = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')

    # Composite layers front-to-back
    out_buffers = {}
    for key in layers[0][0].keys():
        if key == 'shaded':
            accum = composite_buffer(key, layers, background, True)
        else:
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), True)

        # Downscale to framebuffer resolution. Use avg pooling 
        out_buffers[key] = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    return out_buffers

# ==============================================================================================
#  Render UVs
# ==============================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture):

    # clip space transform 
    uv_clip = mesh.v_tex[None, ...]*2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())

    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos)
    assert all_tex.shape[-1] == 6, "Combined kd_ks must be 6 channels"
    return (rast[..., -1:] > 0).float(), all_tex[..., 0:3], all_tex[..., 3:6]