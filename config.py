Dev_model_path = "black-forest-labs/FLUX.1-dev"
Kontext_model_path = "black-forest-labs/FLUX.1-Kontext-dev"
threshold = 0.6 # for dev
last_step = 850
pre_thr = 0.2 #  for kontext
import importlib, types
from importlib import reload
import my_flux.transformer_flux_custom
import gc
import torch
import my_flux.transformer_fluxkontext_custom

def reload_fluxpipeline(pipe):
    import importlib, types
    from importlib import reload
    import my_flux.pipeline_flux_kontext_custom
    reload(my_flux.pipeline_flux_kontext_custom)
    from my_flux.pipeline_flux_kontext_custom import FluxKontextPipeline as NewFluxPipeline
    for method_name in ("process_id", "__call__", "_prepare_latent_image_ids","prepare_latents","encode_prompt", "_get_t5_prompt_embeds"):
        if hasattr(NewFluxPipeline, method_name):
            fn = getattr(NewFluxPipeline, method_name)
            setattr(pipe, method_name, types.MethodType(fn, pipe))
            print(f"patched pipe.{method_name} -> {NewFluxPipeline.__name__}.{method_name}")
    new_call = getattr(NewFluxPipeline, "__call__")
    orig_class = pipe.__class__
    setattr(orig_class, "__call__", new_call)
    print("Reloaded pipeline flux custom classes and patched methods.")

def reload_dev_transformer(pipe):
    reload(my_flux.transformer_flux_custom)
    from my_flux.transformer_flux_custom import FluxTransformer2DModel as NewFluxTransformer
    for method_name in (["forward","__init__"]):
        if hasattr(NewFluxTransformer, method_name):
            new_forward = getattr(NewFluxTransformer, method_name)
            setattr(pipe.transformer, method_name, types.MethodType(new_forward, pipe.transformer))
    from my_flux.transformer_flux_custom import FluxAttnProcessor as NewFluxProcessor
    new_call = getattr(NewFluxProcessor, "__call__")
    for block in pipe.transformer.transformer_blocks:
        if hasattr(block.attn, "processor"):
            orig_class = block.attn.processor.__class__
            setattr(orig_class, "__call__", new_call)
    for single_block in pipe.transformer.single_transformer_blocks:
        if hasattr(single_block.attn, "processor"):
            orig_class = single_block.attn.processor.__class__
            setattr(orig_class, "__call__", new_call)

    from my_flux.transformer_flux_custom import FluxSingleTransformerBlock as NewSingleTransformerBlock
    from my_flux.transformer_flux_custom import FluxTransformerBlock as NewTransformerBlock
    for method_name in (["forward","__init__"]):
        if hasattr(NewSingleTransformerBlock, method_name) and hasattr(NewTransformerBlock, method_name):
            new_forward_single = getattr(NewSingleTransformerBlock, method_name)
            new_forward = getattr(NewTransformerBlock, method_name)
            for single_block in pipe.transformer.single_transformer_blocks:
                setattr(single_block, method_name, types.MethodType(new_forward_single, single_block))
            for block in pipe.transformer.transformer_blocks:
                setattr(block, method_name, types.MethodType(new_forward, block))    

    from my_flux.transformer_flux_custom import FluxAttention as NewFluxAttention
    for method_name in (["forward","__init__"]):
        if hasattr(NewFluxAttention, method_name):
            new_forward = getattr(NewFluxAttention, method_name)
            for single_block in pipe.transformer.single_transformer_blocks:
                setattr(single_block.attn, method_name, types.MethodType(new_forward, single_block.attn))
            for block in pipe.transformer.transformer_blocks:
                setattr(block.attn, method_name, types.MethodType(new_forward, block.attn))

    print("Reloaded transformer flux custom classes and patched methods.")

def reload_kon_transformer(pipe):
    reload(my_flux.transformer_fluxkontext_custom)
    from my_flux.transformer_fluxkontext_custom import FluxTransformer2DModel as NewFluxTransformer
    for method_name in (["forward","__init__"]):
        if hasattr(NewFluxTransformer, method_name):
            new_forward = getattr(NewFluxTransformer, method_name)
            setattr(pipe.transformer, method_name, types.MethodType(new_forward, pipe.transformer))


    from my_flux.transformer_fluxkontext_custom import FluxAttnProcessor as NewFluxProcessor
    new_call = getattr(NewFluxProcessor, "__call__")
    for block in pipe.transformer.transformer_blocks:
        if hasattr(block.attn, "processor"):
            orig_class = block.attn.processor.__class__
            setattr(orig_class, "__call__", new_call)
            # print(f"patched block.attn.processor.__class__.__name__ -> {NewFluxProcessor.__name__}")
    for single_block in pipe.transformer.single_transformer_blocks:
        if hasattr(single_block.attn, "processor"):
            orig_class = single_block.attn.processor.__class__
            setattr(orig_class, "__call__", new_call)

    from my_flux.transformer_fluxkontext_custom import FluxSingleTransformerBlock as NewSingleTransformerBlock
    from my_flux.transformer_fluxkontext_custom import FluxTransformerBlock as NewTransformerBlock
    for method_name in (["forward","__init__"]):
        if hasattr(NewSingleTransformerBlock, method_name) and hasattr(NewTransformerBlock, method_name):
            new_forward_single = getattr(NewSingleTransformerBlock, method_name)
            new_forward = getattr(NewTransformerBlock, method_name)
            for single_block in pipe.transformer.single_transformer_blocks:
                setattr(single_block, method_name, types.MethodType(new_forward_single, single_block))
            for block in pipe.transformer.transformer_blocks:
                setattr(block, method_name, types.MethodType(new_forward, block))    


    from my_flux.transformer_fluxkontext_custom import FluxAttention as NewFluxAttention
    for method_name in (["forward","__init__"]):
        if hasattr(NewFluxAttention, method_name):
            new_forward = getattr(NewFluxAttention, method_name)
            for single_block in pipe.transformer.single_transformer_blocks:
                setattr(single_block.attn, method_name, types.MethodType(new_forward, single_block.attn))
            for block in pipe.transformer.transformer_blocks:
                setattr(block.attn, method_name, types.MethodType(new_forward, block.attn))

def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()
    