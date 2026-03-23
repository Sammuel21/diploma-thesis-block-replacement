import numpy as np

# resolvers

def resolve_target_layers(cfg, n_layers: int, bi_scores: dict):
    reverse = True
    k = min(cfg.k_blocks, len(bi_scores))

    if cfg.selection_strategy == "manual":
        if not cfg.target_layer_idxs:
            raise ValueError("maual selection requires layer selection, no layers to be replaced provided.")
        idxs = cfg.target_layer_idxs

    elif cfg.selection_strategy == "random_k":
        idxs = np.random.choice(list(range(n_layers)), size=k)

    elif cfg.selection_strategy == "top_k_bi":
        reverse = cfg.bi_rank_order == "desc"
        if not bi_scores:
            raise ValueError(f"BI scores required for this strategy, strategy={cfg.selection_strategy}")
        idxs = sorted(bi_scores, key=bi_scores.get, reverse=reverse)[:k]
    
    return idxs


def resolve_replacement_strategy(cfg):
    strategies = ["one_shot", "iterative"]
    strategy = cfg.replacement_strategy
    if strategy not in strategies:
        raise ValueError(f"Unsupported replacement strategy, strategy={strategy}")
    return strategy

