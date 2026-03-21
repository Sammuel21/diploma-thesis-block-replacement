from dataclasses import dataclass
from typing import Literal

@dataclass
class CFG:
    model_id = "HuggingFaceTB/SmolLM2-1.7B"
    target_layer_idx = 3
    seq_len = 128
    batch_size = 2
    calib_split = "train[:5%]"
    eval_split = "validation[:1%]"
    num_calib_batches = 24
    num_eval_batches = 24
    replacement_epochs = 5
    replacement_lr = 1e-3
    replacement_batch_size = 2048
    seed = 21



@dataclass
class MCFG:
    '''Multi-block-CFG'''
    model_id = "HuggingFaceTB/SmolLM2-1.7B"

    target_layer_idx = 3
    target_layer_idxs: tuple[int] = (2, 4, 6)

    selection_strategy: Literal["manual", "random_k", "top_k_bi"] = "manual"
    k_blocks = 3

    swap_mode: Literal["sequential", "one_shot"] = "one_shot"

    seq_len = 128
    batch_size = 2

    calib_split = "train[:5%]"
    eval_split = "validation[:1%]"

    num_calib_batches = 24
    num_eval_batches = 24
    
    replacement_epochs = 5
    replacement_lr = 1e-3
    replacement_batch_size = 2048

    seed = 21