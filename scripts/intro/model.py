from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_id: str, dtype, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Pre jednoduché batching; pri causal LM je to bežná praktická voľba.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)

    # Aby sedel eval s paddingom
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer