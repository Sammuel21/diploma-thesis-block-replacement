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
    model_id = "HuggingFaceTB/SmolLM2-1.7B"

    k_blocks: int = 3
    target_layer_idxs = ()
    selection_strategy: Literal["manual", "first_k", "random_k", "top_k_bi"] = "manual"
    bi_rank_order: Literal["asc", "desc"] = "desc"

    replacement_strategy: Literal["one_shot", "iterative"] = "one_shot"
    replacement_operator: Literal["linear"] = "linear"

    seq_len: int = 128
    batch_size: int = 2
    calib_split: str = "train[:5%]"
    eval_split: str = "validation[:1%]"
    num_calib_batches: int = 24
    num_eval_batches: int = 24
    replacement_epochs: int = 5
    replacement_lr: float = 1e-3
    replacement_batch_size: int = 2048
    seed: int = 21
