import torch

def collect_block_io(model, block_path: str, loader, max_batches: int, device: str):
    block = model.get_submodule(block_path)

    x_chunks = []
    y_chunks = []
    cache = {}

    def pre_hook(module, args):
        # args[0] = input hidden states do MLP
        x = args[0].detach().float().cpu()
        cache["x"] = x

    def post_hook(module, args, output):
        y = output.detach().float().cpu()
        x = cache["x"]

        # [B, T, D] -> [B*T, D]
        x_chunks.append(x.reshape(-1, x.shape[-1]))
        y_chunks.append(y.reshape(-1, y.shape[-1]))

    h1 = block.register_forward_pre_hook(pre_hook)
    h2 = block.register_forward_hook(post_hook)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)

    h1.remove()
    h2.remove()

    X = torch.cat(x_chunks, dim=0)
    Y = torch.cat(y_chunks, dim=0)
    return X, Y