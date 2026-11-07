import numpy as np

# resolvers

def resolve_target_layers(cfg, n_layers: int, bi_scores: dict = None):
    if bi_scores is None:
        bi_scores = {}
    
    k = min(cfg.k_blocks, n_layers)

    if cfg.selection_strategy == "manual":
        if not cfg.target_layer_idxs:
            raise ValueError("maual selection requires layer selection, no layers to be replaced provided.")
        idxs = cfg.target_layer_idxs

    elif cfg.selection_strategy == "random_k":
        rng = np.random.default_rng(cfg.seed)
        idxs = rng.choice(np.arange(n_layers), size=k, replace=False).tolist()

    elif cfg.selection_strategy == "top_k_bi":
        if not bi_scores:
            raise ValueError("BI scores required for this strategy")
        reverse = cfg.bi_rank_order == "desc"
        idxs = sorted(bi_scores, key=bi_scores.get, reverse=reverse)[:k]
    
    return sorted(int(i) for i in idxs)



def resolve_replacement_strategy(cfg):
    strategies = ["one_shot", "iterative"]
    strategy = cfg.replacement_strategy
    if strategy not in strategies:
        raise ValueError(f"Unsupported replacement strategy, strategy={strategy}")
    return strategy

