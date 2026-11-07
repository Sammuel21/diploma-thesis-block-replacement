def count_params(module, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def param_summary(model, target_path: str):
    target_block = model.get_submodule(target_path)
    return {
        "total_params": count_params(model),
        "target_block_params": count_params(target_block),
    }