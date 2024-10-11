
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)

from torch.distributed.fsdp import ShardingStrategy, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

# from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer

# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

non_reentrant_wrapper = partial( 
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

# check_fn = lambda submodule: isinstance(submodule, (LlamaDecoderLayer, MixtralDecoderLayer, MistralDecoderLayer, Phi3DecoderLayer))
# check_fn = lambda submodule: isinstance(submodule, (LlamaDecoderLayer, MixtralDecoderLayer, MistralDecoderLayer))
check_fn = lambda submodule: isinstance(submodule, (LlamaDecoderLayer, MistralDecoderLayer, MixtralDecoderLayer, torch.nn.Embedding))



@dataclass
class fsdp_config:
    mixed_precision: bool=True
    use_fp16: bool=True
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool=True
    pure_bf16: bool = False
    optimizer: str= "AdamW"

def get_model_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper

    model_auto_wrap_policy = partial( # # 用于“冻结”一些函数的参数或关键字参数，生成新的可调用对象。partial允许你创建一个新的函数，该函数会调用指定的函数，但是预先设定了某些参数的值；当你调用这个新的函数时，你可以传入额外的参数，这些参数将与预先设定的参数一起传递给原始函数
        transformer_auto_wrap_policy,
        # transformer_layer_cls=set([LlamaDecoderLayer, MixtralDecoderLayer, MistralDecoderLayer, Phi3DecoderLayer]),
        # transformer_layer_cls=set([LlamaDecoderLayer, MistralDecoderLayer, MixtralDecoderLayer, torch.nn.Embedding]),
        transformer_layer_cls=set([MixtralDecoderLayer, torch.nn.Embedding]),
        # transformer_layer_cls=set([MistralDecoderLayer, torch.nn.Embedding]),
    )
    print("Model auto wrap policy: ",model_auto_wrap_policy)
    return model_auto_wrap_policy

def get_mistral_wrapper():
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper

    mistral_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            MistralDecoderLayer,
        },
    )

    return mistral_auto_wrap_policy

def _auto_wrap_policy(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
    if recurse:
        return True
    if hasattr(module, '_fsdp_wrap'):
        return bool(module._fsdp_wrap)
    return False

def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

# def save_model_checkpoint_(
#     model, 
#     output_dir,
#     rank
# ):
#     """saving model via rank0 cpu streaming and full_state_dict"""

#     with FSDP.state_dict_type(
#         model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
#     ):
#         cpu_state = model.state_dict()

#         print(f"saving process: rank {rank}  done w model state_dict\n")

#     if rank == 0:
#         print(f"--> saving model ...")
#         save_dir = Path.cwd() / output_dir
#         save_dir.mkdir(parents=True, exist_ok=True)
#         save_full_path = str(save_dir) + f"/pytorch_model.bin"
        
#         # save model
#         torch.save(cpu_state, save_full_path)
        
#         print(f"model checkpoint saved at {save_full_path}\n")

def save_model_checkpoint(
    model, 
    output_dir,
    rank
):
    """saving model via rank0 cpu streaming and full_state_dict"""

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()

        print(f"saving process: rank {rank}  done w model state_dict\n")

    if rank == 0:
        print(f"--> saving model ...")
        save_dir = Path.cwd() / output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_full_path = str(save_dir) + "/pytorch_model.bin"
        
        # save model
        torch.save(cpu_state, save_full_path)
        
        print(f"model checkpoint saved at {save_full_path}\n")
