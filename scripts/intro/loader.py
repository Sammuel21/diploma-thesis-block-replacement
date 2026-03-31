import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from scripts.data.datautils import get_wikitext2, get_c4

# datasets

class TokenSequencesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_ids = self.sequences[idx]
        return {
            "input_ids" : input_ids,
            "attention_mask" : torch.ones_like(input_ids)
        }


# datautils interface
# NOTE: naparovat nejako datautils na nas pipeline format s ktorym robim v prototype
# input_ids : [B, S]
# attention_mask: [B, S]

def sequences_from_samples(samples):
    # NOTE: tato funkcia pretransformuje shape inputov z data utils ako [1, S] -> [S]
    sequences = []
    for inp, tar in samples:
        sequence = inp.squeeze(0).clone()
        sequences.append(sequence)

    return sequences


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


# loaders podla datautils backendu

def get_datautils_backend(dataset: str, n_samples, seq_len, model_id: str, tokenizer, seed):
    if dataset.lower() == "wikitext2":
        return get_wikitext2(n_samples, seed, seq_len, model_id, tokenizer)
    
    elif dataset.lower() == "c4":
        return get_c4(n_samples, seed, seq_len, model_id, tokenizer)
    
    raise ValueError(f"Dataset {dataset} not in datautils")



def make_calib_loader(tokenizer, cfg, dataset="Wikitext2"):
    n_samples = cfg.num_calib_batches * cfg.batch_size
    train_samples, val_samples = get_datautils_backend(dataset, n_samples, cfg.seq_len, cfg.model_id, tokenizer, cfg.seed)



def make_eval_loader(tokenizer, cfg, dataset="C4"):
    n_samples = cfg.num_calib_batches * cfg.batch_size
    train_samples, val_samples = get_datautils_backend(dataset, n_samples, cfg.seq_len, cfg.model_id, tokenizer, cfg.seed)