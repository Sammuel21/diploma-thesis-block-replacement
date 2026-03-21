from dataclasses import dataclass
from typing import Literal

@dataclass
class DataConfig:
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    calib_split = "train[:1%]"
    eval_split = "validation[:1%]"
    seq_len = 128
    batch_size = 2
    shuffle_calib = False
    max_calib_batches = 16
    max_eval_batches = 16
