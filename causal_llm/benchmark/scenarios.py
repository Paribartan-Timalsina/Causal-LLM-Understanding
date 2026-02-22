"""Scenario templates and phrasing variations for benchmark question generation.

Each scenario provides a real-world causal story for one of the five canonical
graph structures, along with named variables for question construction.
"""

SCENARIOS = {
    "chain": [
        {
            "story": "A factory increases raw material quality (X). This improves the manufacturing process (Y), which in turn increases product durability (Z).",
            "vars": {"X": "raw material quality", "Y": "manufacturing process quality", "Z": "product durability"},
        },
        {
            "story": "A student studies more hours (X). This improves their understanding of concepts (Y), which leads to higher exam scores (Z).",
            "vars": {"X": "study hours", "Y": "concept understanding", "Z": "exam scores"},
        },
        {
            "story": "Regular exercise (X) improves cardiovascular fitness (Y), which increases overall longevity (Z).",
            "vars": {"X": "exercise", "Y": "cardiovascular fitness", "Z": "longevity"},
        },
        {
            "story": "Increased advertising spending (X) raises brand awareness (Y), which boosts sales revenue (Z).",
            "vars": {"X": "advertising spending", "Y": "brand awareness", "Z": "sales revenue"},
        },
    ],
    "fork": [
        {
            "story": "Socioeconomic status (C) influences both access to quality education (X) and access to healthcare (Z). Education and healthcare do not directly cause each other.",
            "vars": {"C": "socioeconomic status", "X": "education quality", "Z": "healthcare access"},
        },
        {
            "story": "Genetic factors (C) influence both height (X) and shoe size (Z). Height does not cause shoe size or vice versa.",
            "vars": {"C": "genetic factors", "X": "height", "Z": "shoe size"},
        },
        {
            "story": "Temperature (C) affects both ice cream sales (X) and swimming pool attendance (Z). Ice cream sales do not cause pool attendance.",
            "vars": {"C": "temperature", "X": "ice cream sales", "Z": "pool attendance"},
        },
        {
            "story": "Seasonal weather (C) drives both flu rates (X) and heating costs (Z). Flu does not cause higher heating costs.",
            "vars": {"C": "seasonal weather", "X": "flu rates", "Z": "heating costs"},
        },
    ],
    "collider": [
        {
            "story": "Both talent (X) and attractiveness (Z) independently contribute to Hollywood success (M). Talent and attractiveness are unrelated in the general population.",
            "vars": {"X": "talent", "Z": "attractiveness", "M": "Hollywood success"},
        },
        {
            "story": "Both programming skill (X) and communication skill (Z) independently contribute to getting hired at a tech company (M). These two skills are unrelated in the general population.",
            "vars": {"X": "programming skill", "Z": "communication skill", "M": "hiring outcome"},
        },
        {
            "story": "Both engine quality (X) and aerodynamics (Z) independently contribute to a car winning a race (M). Engine quality and aerodynamics are designed independently.",
            "vars": {"X": "engine quality", "Z": "aerodynamics", "M": "race outcome"},
        },
        {
            "story": "Both rainfall (X) and sprinkler use (Z) independently cause wet grass (M). Rainfall and sprinkler use are unrelated.",
            "vars": {"X": "rainfall", "Z": "sprinkler use", "M": "wet grass"},
        },
    ],
    "diamond": [
        {
            "story": "A new CEO (X) changes both the marketing strategy (Y) and the R&D strategy (Z). Both marketing and R&D then influence the company stock price (W).",
            "vars": {"X": "new CEO", "Y": "marketing strategy", "Z": "R&D strategy", "W": "stock price"},
        },
        {
            "story": "A drug (X) affects both liver function (Y) and kidney function (Z). Both liver and kidney function influence overall health outcome (W).",
            "vars": {"X": "drug dosage", "Y": "liver function", "Z": "kidney function", "W": "health outcome"},
        },
        {
            "story": "Government policy (X) affects both education spending (Y) and infrastructure spending (Z). Both types of spending influence economic growth (W).",
            "vars": {"X": "government policy", "Y": "education spending", "Z": "infrastructure spending", "W": "economic growth"},
        },
    ],
    "instrument": [
        {
            "story": "Availability of college scholarships (Z) affects whether a person attends college (X). Both attending college and unobserved ability (U) affect lifetime earnings (Y). Scholarship availability does not directly affect earnings except through college attendance.",
            "vars": {"Z": "scholarship availability", "X": "college attendance", "Y": "lifetime earnings", "U": "unobserved ability"},
        },
        {
            "story": "Parental encouragement (Z) affects amount of reading practice (X). Both reading practice and unobserved cognitive aptitude (U) affect reading test scores (Y). Parental encouragement does not directly affect test scores.",
            "vars": {"Z": "parental encouragement", "X": "reading practice", "Y": "reading test scores", "U": "cognitive aptitude"},
        },
        {
            "story": "Quarter of birth (Z) affects years of schooling (X). Both schooling and unobserved innate ability (U) affect wages (Y). Quarter of birth does not directly affect wages.",
            "vars": {"Z": "quarter of birth", "X": "years of schooling", "Y": "wages", "U": "innate ability"},
        },
    ],
}

# Phrasing variations keyed by reasoning level.
# make_question() picks one via q_id to ensure unique text across repeated scenarios.

L1_OBSERVE_PHRASES = [
    lambda v: f"We observe that {v} is high. What do we expect for",
    lambda v: f"Suppose we notice that {v} is at a high level. What should we predict for",
    lambda v: f"Given that {v} is observed to be high, what is the likely outcome for",
    lambda v: f"If we see that {v} is high, what can we infer about",
    lambda v: f"Imagine we measure {v} and find it is high. What follows for",
]

L2_INTERVENE_PHRASES = [
    lambda v: f"If we intervene and force {v} to be high (do({v}=high)), what happens to",
    lambda v: f"Suppose we externally set {v} to a high value. What is the effect on",
    lambda v: f"If we perform an intervention do({v}=high), what change do we expect in",
    lambda v: f"Consider forcibly making {v} high, breaking its natural causes. What happens to",
    lambda v: f"If an experimenter manipulates {v} to be high, what is the causal impact on",
]

L3_CF_PHRASES = [
    lambda x, z: f"We observed {x}=low and {z}=low. If {x} had been high instead, what would {z} have been?",
    lambda x, z: f"Given that {x} was low and {z} was low, suppose {x} had been high. What would {z} be?",
    lambda x, z: f"In a world where {x}=low led to {z}=low, what would {z} have been if {x} were high?",
    lambda x, z: f"Counterfactual: we saw {x}=low, {z}=low. Had {x} been high, what about {z}?",
    lambda x, z: f"{x} was low and {z} ended up low. If we could go back and set {x} to high, what would {z} be?",
]

CONTEXT_PREFIXES = [
    "",
    "Consider this situation carefully. ",
    "Think about the causal relationships here. ",
    "Pay attention to the direction of causation. ",
    "Analyze the following causal scenario. ",
]

FEW_SHOT_EXAMPLES = {
    "L1": {
        "scenario": "Sunlight (X) causes plants to grow taller (Y).",
        "question": "We observe lots of sunlight. What do we expect for plant height?",
        "choices": {"A": "Taller", "B": "Shorter", "C": "No change", "D": "Cannot tell"},
        "answer": "A",
        "reasoning": "Sunlight causes growth, so observing sunlight predicts taller plants.",
    },
    "L2": {
        "scenario": "Weather (C) causes both ice cream sales (X) and drowning rates (Y). X does not cause Y.",
        "question": "If we intervene and force high ice cream sales do(X=high), what happens to drowning rates?",
        "choices": {"A": "Increases", "B": "No effect", "C": "Decreases", "D": "Doubles"},
        "answer": "B",
        "reasoning": "Intervening on X breaks the C->X link. Since X does not cause Y, Y is unaffected.",
    },
    "L3": {
        "scenario": "Studying (X) causes good grades (Y). A student studied hard and got good grades.",
        "question": "If the student had NOT studied, would they have gotten good grades?",
        "choices": {"A": "Yes", "B": "No", "C": "Same grades", "D": "Better grades"},
        "answer": "B",
        "reasoning": "Counterfactual: removing the cause (studying) removes the effect (good grades).",
    },
}
