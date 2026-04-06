import torch
from PIL import Image


def is_hip():
    return torch.version.hip is not None


def use_compile(pipeline, resolution: int = 512):
    # Compile the compute-intensive portions of the model: denoising transformer / decoder
    # Compile transformer w/o fullgraph and cudagraphs if cache-dit is enabled.
    # The cache-dit relies heavily on dynamic Python operations to maintain the cache_context, 
    # so it is necessary to introduce graph breaks at appropriate positions to be compatible 
    # with torch.compile. Thus, we compile the transformer with `max-autotune-no-cudagraphs` 
    # mode if cache-dit is enabled. Otherwise, we compile with `max-autotune` mode.
    is_cached = getattr(pipeline.transformer, "_is_cached", False)
    # For AMD MI300X w/ the AITER kernels, the default dynamic=None is not working as expected, giving black results.
    # Therefore, we use dynamic=True for AMD only. This leads to a small perf penalty, but should be fixed eventually. 
    pipeline.transformer = torch.compile(
        pipeline.transformer, 
        mode="max-autotune" if not is_cached else "max-autotune-no-cudagraphs", 
        fullgraph=(True if not is_cached else False), 
        dynamic=True if is_hip() else None
    )
    pipeline.vae.decode = torch.compile(
        pipeline.vae.decode, mode="max-autotune", fullgraph=True, dynamic=True if is_hip() else None
    )

    # warmup for a few iterations (`num_inference_steps` shouldn't matter)
    res = resolution
    size = (res, res)
    input_kwargs = {
        "image": Image.new("RGB", size=size),
        "num_inference_steps": 3,
        "infer_mode": "AL",
        "num_in_batch": 6,
        "height": res,
        "width": res,
        "normal_imgs": [[Image.new("RGBA", size=size) for _ in range(6)]],
        # "position_imgs": [[Image.new("RGBA", size=size) for _ in range(6)]],
        # "mask_imgs": [[Image.new("RGBA", size=size) for _ in range(6)]],
        "ref_img": [[Image.new("RGBA", size=size)]],
        "prompt_embeds": torch.zeros((1, 128, 4096), dtype=torch.bfloat16, device=pipeline.device),
        "pooled_prompt_embeds": torch.zeros((1, 768), dtype=torch.bfloat16, device=pipeline.device),
    }
    for _ in range(3):
        pipeline(**input_kwargs).images[0]

    return pipeline

def load_flux_lora_weights(transformer, lora_path, alpha):
    """
        Usage:
            pipe.transformer = load_flux_lora_weights(pipe.transformer, "./checkpoint/lora/FLUXPro1.1.safetensors", 0.7)
    """
    from safetensors.torch import load_file
    from diffusers import StableDiffusionPipeline
    
    device = transformer.device
    # load LoRA weight from .safetensors
    state_dict = load_file(lora_path)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        layer_infos = key.split('.')[1:-2] # already begin with `transformer`, exclude `lora_X` and `weight`
        curr_layer = transformer
        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            if temp_name == "transformer":
                temp_name = layer_infos.pop(0)
                continue
            # print(f"layer_infos: {layer_infos}")
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(layer_infos) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_A" in key:
            pair_keys.append(key.replace("lora_A", "lora_B"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_B", "lora_A"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(device=device, dtype=torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(device=device, dtype=torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(device=device, dtype=torch.float32)
            weight_down = state_dict[pair_keys[1]].to(device=device, dtype=torch.float32)
            print(f"weight_up: {weight_up.shape}, weight_down: {weight_down.shape}")
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

    return transformer

if __name__ == "__main__":
    
    from diffusers import FluxImg2ImgPipeline
    from diffusers.utils import load_image
    from huggingface_hub import hf_hub_download

    device = "cuda"
    pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.transformer = load_flux_lora_weights(pipe.transformer, "./checkpoint/lora/FLUXPro1.1.safetensors", 0.7)

    ########## GENERATE ##########
    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    ref_image = load_image(url).resize((1024, 1024))

    prompt = "Photo of a lowpoly fantasy house from warcraft game, lawn. , aidmafluxpro1.1"

    images = pipe(
        prompt=prompt,
        image=ref_image,
        height=1024,
        width=1024,
        num_inference_steps=30,
        strength=1.0,
        guidance_scale=3.5,
        generator=torch.Generator().manual_seed(42)
    ).images[0]

    images.save("test_load_lora.png")