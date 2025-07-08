"""
This file specifies how MLC's RWKV7 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

from ...loader import ExternMapping
from ...quantization import Quantization
from .rwkv7_model import RWKV7_ForCasualLM, RWKV7Config


def huggingface(model_config: RWKV7Config, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : RWKV7Config
        The configuration of the RWKV7 model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = RWKV7_ForCasualLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params = model.export_tvm(  # pylint: disable=unbalanced-tuple-unpacking
        spec=model.get_default_spec()
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            hf_name = mlc_name.replace("blocks", "layers").replace("attention", "attn").replace("feed_forward", "ffn")
            hf_name = hf_name.replace("ln_out", "norm").replace("ln1", "attn_norm").replace("ln2", "ffn_norm")
            hf_name = hf_name.replace("ln_x", "g_norm").replace("pre_ln", "pre_norm")
            hf_name = hf_name.replace("attn.output", "attn.o_proj").replace("attn.receptance", "attn.r_proj")
            hf_name = hf_name.replace("attn.key", "attn.k_proj").replace("attn.value", "attn.v_proj")
            hf_name = hf_name.replace("head", "lm_head")
            hf_name = hf_name.replace("2.bias", "_lora.lora.2.bias")
            hf_name = hf_name.replace("2.weight", "_lora.lora.2.weight")
            hf_name = hf_name.replace("1.weight", "_lora.lora.0.weight")
            mapping.add_mapping(
                mlc_name,
                [hf_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    return mapping
