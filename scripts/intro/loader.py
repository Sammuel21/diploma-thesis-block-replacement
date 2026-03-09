from torch.utils.data import DataLoader
from datasets import load_dataset

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