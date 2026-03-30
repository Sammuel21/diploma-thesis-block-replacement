import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# datasets

class TokenSequences(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        input_ids = self.seqs[idx]
        return {
            "input_ids" : input_ids,
            "attention_mask" : torch.ones_like(input_ids)
        }


# datautils interface
# NOTE: naparovat nejako datautils na nas pipeline format s ktorym robim v prototype
# input_ids : [B, S]
# attention_mask: [B, S]



# loaders

def make_loader(tokenizer, split: str, seq_len: int, batch_size: int):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    ds = ds.filter(lambda x: x["text"] is not None and x["text"].strip() != "")

    def tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )

    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def make_calib_loader(tokenizer, cfg):
    n_samples = cfg.num_calib_batches * cfg.batch_size


def make_eval_loader(tokenizer, cfg):
    n_samples = cfg.num_calib_batches * cfg.batch_size
