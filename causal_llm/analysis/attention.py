"""Attention pattern extraction and causal keyword analysis."""

import numpy as np
import torch

from ..config import MAX_SEQ_LEN


@torch.no_grad()
def extract_attention(model, tokenizer, prompt, device, max_len=MAX_SEQ_LEN):
    """Extract attention weights from all layers/heads for a single prompt.

    Args:
        model: HuggingFace causal LM
        tokenizer: Corresponding tokenizer
        prompt: string prompt
        device: torch device
        max_len: maximum sequence length

    Returns:
        tuple: (attention_weights, tokens)
            attention_weights: np.array of shape (n_layers, n_heads, seq_len, seq_len)
            tokens: list of token strings
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_len).to(device)
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attn_array = np.stack([a[0].cpu().numpy() for a in attentions], axis=0)

    return attn_array, tokens


def analyze_causal_keyword_attention(attention_weights, tokens, keywords=None):
    """Analyze attention directed toward causal keywords by layer.

    Args:
        attention_weights: np.array (n_layers, n_heads, seq_len, seq_len)
        tokens: list of token strings
        keywords: list of keyword strings to track. Defaults to common causal terms.

    Returns:
        dict: {keyword: np.array of mean attention per layer}
    """
    if keywords is None:
        keywords = ["cause", "causes", "effect", "intervene", "do(",
                     "counterfactual", "if", "had", "would"]

    token_strs = [t.lower().replace("Ġ", "").replace("▁", "") for t in tokens]
    n_layers = attention_weights.shape[0]

    results = {}
    for kw in keywords:
        kw_lower = kw.lower()
        kw_positions = [i for i, t in enumerate(token_strs) if kw_lower in t]
        if not kw_positions:
            continue

        layer_means = []
        for layer_idx in range(n_layers):
            attn = attention_weights[layer_idx]
            kw_attn = attn[:, :, kw_positions].mean()
            layer_means.append(float(kw_attn))

        results[kw] = np.array(layer_means)

    return results
