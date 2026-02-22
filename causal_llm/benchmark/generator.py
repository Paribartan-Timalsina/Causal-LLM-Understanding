"""Benchmark generation: question construction, shuffling, and deduplication."""

import hashlib
import random
from collections import Counter, defaultdict

from .scenarios import (
    SCENARIOS,
    L1_OBSERVE_PHRASES,
    L2_INTERVENE_PHRASES,
    L3_CF_PHRASES,
    CONTEXT_PREFIXES,
)
from ..config import SEED, NUM_QUESTIONS_PER_CELL


def make_question(scenario, graph_type, level, q_id):
    """Generate a single causal reasoning question with ground-truth answer.

    Uses q_id to select phrasing variations so repeated scenarios produce
    unique text.
    """
    s = scenario
    v = s["vars"]
    prefix = CONTEXT_PREFIXES[q_id % len(CONTEXT_PREFIXES)]
    story = prefix + s["story"]

    if graph_type == "chain":
        X, Y, Z = v["X"], v["Y"], v["Z"]
        if level == "L1":
            qtext = L1_OBSERVE_PHRASES[q_id % len(L1_OBSERVE_PHRASES)](X) + f" {Z}?"
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": qtext,
                "choices": {
                    "A": f"{Z} is likely higher than average",
                    "B": f"{Z} is likely lower than average",
                    "C": f"{Z} is unaffected",
                    "D": f"Cannot determine from the given information",
                },
                "answer": "A",
                "explanation": f"In a chain X->Y->Z, observing X=high implies Y=high implies Z=high.",
            }
        elif level == "L2":
            qtext = L2_INTERVENE_PHRASES[q_id % len(L2_INTERVENE_PHRASES)](X) + f" {Z}?"
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": qtext,
                "choices": {
                    "A": f"{Z} increases (causal effect propagates through {Y})",
                    "B": f"{Z} is unaffected because the intervention breaks the chain",
                    "C": f"{Z} decreases due to the intervention",
                    "D": f"The effect depends on unobserved confounders",
                },
                "answer": "A",
                "explanation": f"Intervening on X in a chain still propagates: do(X=high)->Y=high->Z=high.",
            }
        else:
            qtext = L3_CF_PHRASES[q_id % len(L3_CF_PHRASES)](X, Z)
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": qtext,
                "choices": {
                    "A": f"{Z} would have been higher",
                    "B": f"{Z} would have remained low",
                    "C": f"{Z} would have been lower",
                    "D": f"Cannot determine without knowing the exact structural equations",
                },
                "answer": "A",
                "explanation": f"Counterfactual: setting X=high would propagate through Y to make Z higher.",
            }

    elif graph_type == "fork":
        C, X, Z = v["C"], v["X"], v["Z"]
        if level == "L1":
            qtext = L1_OBSERVE_PHRASES[q_id % len(L1_OBSERVE_PHRASES)](X) + f" {Z}?"
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": qtext,
                "choices": {
                    "A": f"{Z} is likely higher than average",
                    "B": f"{Z} is likely lower than average",
                    "C": f"{Z} is completely unaffected by {X}",
                    "D": f"{Z} is inversely related to {X}",
                },
                "answer": "A",
                "explanation": f"In a fork C->X, C->Z: observing X=high suggests C=high, which suggests Z=high. Correlation through the confounder.",
            }
        elif level == "L2":
            qtext = L2_INTERVENE_PHRASES[q_id % len(L2_INTERVENE_PHRASES)](X) + f" {Z}?"
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": qtext,
                "choices": {
                    "A": f"{Z} increases because {X} and {Z} are correlated",
                    "B": f"{Z} is unaffected because do({X}) severs the link from {C} to {X}",
                    "C": f"{Z} decreases due to a compensatory mechanism",
                    "D": f"{Z} doubles because of the intervention",
                },
                "answer": "B",
                "explanation": f"Intervening do(X=high) cuts the edge C->X. Since X does not cause Z (only C does), Z is unaffected.",
            }
        else:
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": L3_CF_PHRASES[q_id % len(L3_CF_PHRASES)](X, Z),
                "choices": {
                    "A": f"{Z} would have been low because {X} and {Z} are correlated",
                    "B": f"{Z} would still be high because {C}=high still causes {Z}=high directly",
                    "C": f"{Z} would have been average",
                    "D": f"Cannot determine",
                },
                "answer": "B",
                "explanation": f"Counterfactual do(X=low): Z is caused by C, not X. Since C=high is still true, Z remains high.",
            }

    elif graph_type == "collider":
        X, Z, M = v["X"], v["Z"], v["M"]
        _collider_L1 = [
            f"In the general population (without conditioning on {M}), if we observe {X} is high, what do we expect for {Z}?",
            f"Without selecting on {M}, suppose we see {X} is high. What can we say about {Z}?",
            f"Ignoring {M} entirely, we notice {X} is high. Does that tell us anything about {Z}?",
            f"Before knowing anything about {M}, we observe {X} is high. What follows for {Z}?",
            f"If we simply observe {X}=high in the population (without conditioning on {M}), what about {Z}?",
        ]
        _collider_L2 = [
            f"Among cases where {M} is positive (conditioning on the collider), we observe {X} is high. If we intervene and set do({X}=low), what happens to {Z} among these cases?",
            f"Conditioning on {M}=positive, we see {X} is high. If we forcibly set {X} to low, what is the causal effect on {Z}?",
            f"Given {M}=positive, suppose we perform do({X}=low). How does {Z} change?",
            f"We select cases with {M} positive and observe {X}=high. What is the causal impact on {Z} if we intervene to make {X} low?",
            f"After conditioning on collider {M}, we intervene on {X}. What happens to {Z}?",
        ]
        if level == "L1":
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": _collider_L1[q_id % len(_collider_L1)],
                "choices": {
                    "A": f"{Z} is likely higher",
                    "B": f"{Z} is likely lower",
                    "C": f"{Z} is unrelated to {X} (no information gained)",
                    "D": f"{Z} depends on {M}",
                },
                "answer": "C",
                "explanation": f"In a collider X->M<-Z, X and Z are marginally independent. Observing X tells us nothing about Z.",
            }
        elif level == "L2":
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": _collider_L2[q_id % len(_collider_L2)],
                "choices": {
                    "A": f"{Z} increases because removing {X}'s contribution means {Z} must explain {M}",
                    "B": f"{Z} is unaffected because {X} does not cause {Z}",
                    "C": f"{Z} decreases proportionally",
                    "D": f"The question is ill-defined",
                },
                "answer": "B",
                "explanation": f"Even when conditioning on collider M, intervening do(X) has no causal effect on Z because there is no causal path X->Z.",
            }
        else:
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": L3_CF_PHRASES[q_id % len(L3_CF_PHRASES)](X, Z),
                "choices": {
                    "A": f"{Z} would have been higher to compensate and keep {M} positive",
                    "B": f"{Z} would have remained low (no causal connection)",
                    "C": f"{Z} would have been lower",
                    "D": f"{M} would have changed but {Z} stays the same",
                },
                "answer": "B",
                "explanation": f"Counterfactual: X does not cause Z. Changing X would affect M, but Z is determined by its own mechanisms independently of X.",
            }

    elif graph_type == "diamond":
        X, Y, Z, W = v["X"], v["Y"], v["Z"], v["W"]
        if level == "L1":
            qtext = L1_OBSERVE_PHRASES[q_id % len(L1_OBSERVE_PHRASES)](X) + f" {W}?"
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": qtext,
                "choices": {
                    "A": f"{W} is likely higher (effect flows through both {Y} and {Z})",
                    "B": f"{W} is unaffected because the two paths cancel out",
                    "C": f"{W} is lower because of competing pathways",
                    "D": f"Only {Y} is affected, not {W}",
                },
                "answer": "A",
                "explanation": f"In a diamond X->{{Y,Z}}->W, X=high implies Y=high and Z=high, both contributing to W=high.",
            }
        elif level == "L2":
            _diamond_L2 = [
                f"If we intervene on {Y} alone (do({Y}=high)) without changing {X}, what happens to {W}?",
                f"Suppose we forcibly set {Y} to high while leaving {X} untouched. What is the effect on {W}?",
                f"If an experimenter sets do({Y}=high) but does not manipulate {X}, what happens to {W}?",
                f"Consider an intervention that makes {Y} high without affecting {X}. What about {W}?",
                f"We perform do({Y}=high) independently of {X}. How does {W} change?",
            ]
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": _diamond_L2[q_id % len(_diamond_L2)],
                "choices": {
                    "A": f"{W} increases because {Y} directly causes {W}",
                    "B": f"{W} is unaffected because we did not change {X}",
                    "C": f"{W} increases fully as if {X} were high",
                    "D": f"{Z} also changes, amplifying the effect on {W}",
                },
                "answer": "A",
                "explanation": f"do(Y=high) directly affects W through Y->W. Z is unchanged (no edge Y->Z). W increases partially.",
            }
        else:
            qtext = L3_CF_PHRASES[q_id % len(L3_CF_PHRASES)](X, W)
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": qtext,
                "choices": {
                    "A": f"{W} would have been higher (both pathways contribute positively)",
                    "B": f"{W} would have stayed low (pathways cancel out)",
                    "C": f"Only one pathway would activate, so {W} increases slightly",
                    "D": f"Cannot determine without more information",
                },
                "answer": "A",
                "explanation": f"Counterfactual: X=high would increase both Y and Z, both increasing W.",
            }

    elif graph_type == "instrument":
        Zv, X, Y, U = v["Z"], v["X"], v["Y"], v["U"]
        if level == "L1":
            qtext = L1_OBSERVE_PHRASES[q_id % len(L1_OBSERVE_PHRASES)](X) + f" {Y}?"
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": qtext,
                "choices": {
                    "A": f"{Y} is likely higher (but this confounds the causal effect with {U})",
                    "B": f"{Y} is unrelated to {X}",
                    "C": f"{Y} is lower",
                    "D": f"{Y} depends only on {Zv}",
                },
                "answer": "A",
                "explanation": f"Observing X=high is informative about Y through both X->Y and U->X,U->Y paths. The association overestimates the causal effect.",
            }
        elif level == "L2":
            _iv_L2 = [
                f"To estimate the causal effect of {X} on {Y} without confounding from {U}, which approach is valid?",
                f"How can we isolate the true causal effect of {X} on {Y}, given the confounding by {U}?",
                f"What method removes confounding by {U} when estimating {X}'s effect on {Y}?",
                f"Given that {U} confounds {X} and {Y}, which strategy correctly identifies the causal effect?",
                f"We want the causal (not confounded) effect of {X} on {Y}. What should we do about {U}?",
            ]
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": _iv_L2[q_id % len(_iv_L2)],
                "choices": {
                    "A": f"Simply compare {Y} among those with high vs low {X}",
                    "B": f"Use {Zv} as an instrumental variable (it affects {X} but not {Y} directly)",
                    "C": f"Condition on {U} to block the confounding path",
                    "D": f"The causal effect cannot be estimated from observational data",
                },
                "answer": "B",
                "explanation": f"{Zv} is a valid instrument: it affects X, is independent of U, and has no direct effect on Y.",
            }
        else:
            _iv_L3 = [
                f"Person A has {Zv}=low, so {X}=low and {Y}=low. If {Zv} had been high (so {X}=high), what would {Y} have been?",
                f"Suppose {Zv} was low, leading to {X}=low and {Y}=low. Had {Zv} been high instead, what would {Y} be?",
                f"Counterfactual: {Zv}=low caused {X}=low and {Y}=low. If {Zv} had been high, what would {Y} be?",
                f"Someone with {Zv}=low ended up with {X}=low, {Y}=low. What if {Zv} had been high?",
                f"If we could change {Zv} from low to high for this individual, how would {Y} change?",
            ]
            return {
                "id": q_id,
                "graph": graph_type,
                "level": level,
                "scenario": story,
                "question": _iv_L3[q_id % len(_iv_L3)],
                "choices": {
                    "A": f"{Y} would have increased by exactly the causal effect of {X} on {Y}",
                    "B": f"{Y} would have increased by the full observed correlation between {X} and {Y}",
                    "C": f"{Y} would be unchanged because {Zv} does not directly affect {Y}",
                    "D": f"{Y} would decrease",
                },
                "answer": "A",
                "explanation": f"Counterfactual through instrument: changing Z changes X, which causally changes Y by the true causal effect (not the confounded association).",
            }

    raise ValueError(f"Unknown graph_type={graph_type!r} or level={level!r}")


def _shuffle_choices_inplace(q, rng):
    """Shuffle answer labels A/B/C/D to eliminate label-frequency bias."""
    labels = ["A", "B", "C", "D"]
    perm = labels.copy()
    rng.shuffle(perm)
    mapping = {labels[i]: perm[i] for i in range(4)}
    new_choices = {mapping[k]: v for k, v in q["choices"].items()}
    q["choices"] = {k: new_choices[k] for k in labels}
    q["answer"] = mapping[q["answer"]]
    return q


def _question_fingerprint(q):
    """Stable fingerprint to detect duplicate questions (text-only)."""
    s = q["scenario"] + "\n" + q["question"] + "\n" + "\n".join(
        [f"{k}:{q['choices'][k]}" for k in ["A", "B", "C", "D"]]
    )
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def generate_benchmark(num_per_cell=None, seed=None):
    """Generate the full causal reasoning benchmark.

    Args:
        num_per_cell: Number of questions per (graph_type, level) cell.
                      Defaults to NUM_QUESTIONS_PER_CELL from config.
        seed: Random seed. Defaults to SEED from config.

    Returns:
        list[dict]: List of question dictionaries, shuffled.
    """
    if num_per_cell is None:
        num_per_cell = NUM_QUESTIONS_PER_CELL
    if seed is None:
        seed = SEED

    benchmark = []
    q_id = 0
    rng = random.Random(seed)
    seen_fps = set()
    max_tries = 50

    for graph_type, scenarios in SCENARIOS.items():
        for level in ["L1", "L2", "L3"]:
            for _ in range(num_per_cell):
                tries = 0
                while True:
                    scenario = rng.choice(scenarios)
                    q = make_question(scenario, graph_type, level, q_id)
                    q_rng = random.Random(seed + 10_000 * q_id + 17)
                    q = _shuffle_choices_inplace(q, q_rng)
                    fp = _question_fingerprint(q)
                    if fp not in seen_fps:
                        seen_fps.add(fp)
                        benchmark.append(q)
                        q_id += 1
                        break
                    tries += 1
                    if tries >= max_tries:
                        benchmark.append(q)
                        q_id += 1
                        break

    random.Random(seed).shuffle(benchmark)
    return benchmark
