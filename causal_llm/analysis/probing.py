"""Linear probing and hidden state extraction for internal representation analysis."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MAX_SEQ_LEN, PROBE_CV_FOLDS


@torch.no_grad()
def extract_hidden_states(model, tokenizer, prompts, device, max_len=MAX_SEQ_LEN):
    """Extract hidden states from all layers for a list of prompts.

    Uses forward hooks to capture activations from every transformer layer.
    Takes the last token's hidden state as the representation for each prompt.

    Args:
        model: HuggingFace causal LM
        tokenizer: Corresponding tokenizer
        prompts: list of prompt strings
        device: torch device
        max_len: maximum sequence length

    Returns:
        dict: {layer_idx: np.array of shape (n_prompts, hidden_dim)}
    """
    model.eval()
    hidden_states_by_layer = {}

    try:
        layers = model.transformer.h
    except AttributeError:
        try:
            layers = model.model.layers
        except AttributeError:
            raise ValueError("Cannot determine transformer layers for this model architecture.")

    num_layers = len(layers)
    for i in range(num_layers):
        hidden_states_by_layer[i] = []

    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            hidden_states_by_layer[layer_idx].append(h[:, -1, :].detach().cpu().numpy())
        return hook_fn

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    try:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=max_len).to(device)
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    result = {}
    for layer_idx in hidden_states_by_layer:
        result[layer_idx] = np.concatenate(hidden_states_by_layer[layer_idx], axis=0)

    return result


def run_probing(hidden_states, labels, task_name, cv_folds=PROBE_CV_FOLDS,
                n_control_repeats=5):
    """Train linear probes at each layer with proper CV hygiene.

    For each layer, fits a logistic regression probe using StratifiedKFold CV.
    Also runs shuffled-label control probes to compute selectivity.

    Args:
        hidden_states: dict {layer_idx: np.array (n_samples, hidden_dim)}
        labels: np.array of integer labels (n_samples,)
        task_name: string name for logging
        cv_folds: number of CV folds
        n_control_repeats: number of shuffled-label control runs

    Returns:
        dict: {
            'actual': {layer: mean_accuracy},
            'control': {layer: mean_control_accuracy},
            'selectivity': {layer: actual - control},
        }
    """
    actual_accs = {}
    control_accs = {}

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"  Skipping probing for '{task_name}': fewer than 2 classes.")
        return {"actual": {}, "control": {}, "selectivity": {}}

    min_class_count = min(np.sum(labels == c) for c in unique_labels)
    effective_folds = min(cv_folds, min_class_count)
    if effective_folds < 2:
        print(f"  Skipping probing for '{task_name}': not enough samples per class.")
        return {"actual": {}, "control": {}, "selectivity": {}}

    for layer_idx in sorted(hidden_states.keys()):
        X = hidden_states[layer_idx]
        y = labels

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)),
        ])

        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
        fold_accs = []
        for train_idx, test_idx in skf.split(X, y):
            pipe.fit(X[train_idx], y[train_idx])
            fold_accs.append(pipe.score(X[test_idx], y[test_idx]))
        actual_accs[layer_idx] = np.mean(fold_accs)

        ctrl_scores = []
        for rep in range(n_control_repeats):
            y_shuffled = np.random.RandomState(rep).permutation(y)
            fold_accs_ctrl = []
            for train_idx, test_idx in skf.split(X, y_shuffled):
                pipe.fit(X[train_idx], y_shuffled[train_idx])
                fold_accs_ctrl.append(pipe.score(X[test_idx], y_shuffled[test_idx]))
            ctrl_scores.append(np.mean(fold_accs_ctrl))
        control_accs[layer_idx] = np.mean(ctrl_scores)

    selectivity = {
        k: actual_accs[k] - control_accs[k]
        for k in actual_accs
    }

    return {
        "actual": actual_accs,
        "control": control_accs,
        "selectivity": selectivity,
    }
