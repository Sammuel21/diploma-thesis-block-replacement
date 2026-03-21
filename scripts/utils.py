import numpy as np

def resolve_raget_layers(cfg, n_layers: int, bi_scores: dict):
    if cfg.target_layer_idxs:
        idxs = list(cfg.target_layer_idxs)

    elif cfg.target_layer_idx is not None:
        idxs = [cfg.target_layer_idx]

    elif cfg.selectionm_strategy == "random_k":
        idxs = np.random.choice(list(range(n_layers)), size=cfg.k_blocks)

    elif cfg.selection_strategy == "top_k_bi":
        if not bi_scores:
            raise ValueError(f"BI scores required for this strategy, strategy={cfg.selection_strategy}")
        idxs = sorted(bi_scores, key=bi_scores.get, reverse=True)[:cfg.k_blocks]
    
    return idxs