"""Realistic causal-story templates for each graph type.

Each scenario has a `story` string and a `vars` dict mapping graph-variable
names (e.g. 'X', 'Y') to human-readable concept names. The benchmark generator
substitutes these into the question text.
"""

from __future__ import annotations

SCENARIOS = {
    'chain': [
        {
            'story': 'A factory increases raw material quality (X). This improves the manufacturing process (Y), which in turn increases product durability (Z).',
            'vars': {'X': 'raw material quality', 'Y': 'manufacturing process quality', 'Z': 'product durability'},
        },
        {
            'story': 'A student studies more hours (X). This improves their understanding of concepts (Y), which leads to higher exam scores (Z).',
            'vars': {'X': 'study hours', 'Y': 'concept understanding', 'Z': 'exam scores'},
        },
        {
            'story': 'Regular exercise (X) improves cardiovascular fitness (Y), which increases overall longevity (Z).',
            'vars': {'X': 'exercise', 'Y': 'cardiovascular fitness', 'Z': 'longevity'},
        },
        {
            'story': 'Increased advertising spending (X) raises brand awareness (Y), which boosts sales revenue (Z).',
            'vars': {'X': 'advertising spending', 'Y': 'brand awareness', 'Z': 'sales revenue'},
        },
        {
            'story': 'Higher temperatures (X) melt more glaciers (Y), which raises sea levels (Z).',
            'vars': {'X': 'temperature', 'Y': 'glacier melting', 'Z': 'sea level'},
        },
        {
            'story': 'More practice (X) builds better technique (Y), which improves competition results (Z).',
            'vars': {'X': 'practice hours', 'Y': 'technique quality', 'Z': 'competition results'},
        },
        {
            'story': 'Heavier rainfall (X) saturates soil (Y), which reduces crop yield (Z).',
            'vars': {'X': 'rainfall', 'Y': 'soil saturation', 'Z': 'crop yield'},
        },
    ],
    'fork': [
        {
            'story': 'Socioeconomic status (C) influences both access to quality education (X) and access to healthcare (Z). Education and healthcare do not directly cause each other.',
            'vars': {'C': 'socioeconomic status', 'X': 'education quality', 'Z': 'healthcare access'},
        },
        {
            'story': 'Genetic factors (C) influence both height (X) and shoe size (Z). Height does not cause shoe size or vice versa.',
            'vars': {'C': 'genetic factors', 'X': 'height', 'Z': 'shoe size'},
        },
        {
            'story': 'Temperature (C) affects both ice cream sales (X) and swimming pool attendance (Z). Ice cream sales do not cause pool attendance.',
            'vars': {'C': 'temperature', 'X': 'ice cream sales', 'Z': 'pool attendance'},
        },
        {
            'story': 'Seasonal weather (C) drives both flu rates (X) and heating costs (Z). Flu does not cause higher heating costs.',
            'vars': {'C': 'seasonal weather', 'X': 'flu rates', 'Z': 'heating costs'},
        },
        {
            'story': 'Ambient humidity (C) controls both perceived temperature (X) and mold growth (Z). Neither directly causes the other.',
            'vars': {'C': 'humidity', 'X': 'perceived temperature', 'Z': 'mold growth'},
        },
        {
            'story': 'An economic recession (C) affects both unemployment rates (X) and suicide rates (Z). Unemployment does not directly cause suicide in this context.',
            'vars': {'C': 'economic recession', 'X': 'unemployment rate', 'Z': 'suicide rate'},
        },
        {
            'story': 'Age (C) influences both grey hair (X) and reaction time (Z). Neither directly affects the other.',
            'vars': {'C': 'age', 'X': 'grey hair', 'Z': 'reaction time'},
        },
    ],
    'collider': [
        {
            'story': 'Both talent (X) and attractiveness (Z) independently contribute to Hollywood success (M). Talent and attractiveness are unrelated in the general population.',
            'vars': {'X': 'talent', 'Z': 'attractiveness', 'M': 'Hollywood success'},
        },
        {
            'story': 'Both programming skill (X) and communication skill (Z) independently contribute to getting hired at a tech company (M). These two skills are unrelated in the general population.',
            'vars': {'X': 'programming skill', 'Z': 'communication skill', 'M': 'hiring outcome'},
        },
        {
            'story': 'Both engine quality (X) and aerodynamics (Z) independently contribute to a car winning a race (M). Engine quality and aerodynamics are designed independently.',
            'vars': {'X': 'engine quality', 'Z': 'aerodynamics', 'M': 'race outcome'},
        },
        {
            'story': 'Both rainfall (X) and sprinkler use (Z) independently cause wet grass (M). Rainfall and sprinkler use are unrelated.',
            'vars': {'X': 'rainfall', 'Z': 'sprinkler use', 'M': 'wet grass'},
        },
        {
            'story': 'Both physical fitness (X) and technical skill (Z) independently contribute to athletic performance (M). Fitness and skill are not correlated in the general population.',
            'vars': {'X': 'physical fitness', 'Z': 'technical skill', 'M': 'athletic performance'},
        },
        {
            'story': 'Both sweetness (X) and acidity (Z) independently contribute to the overall flavor score (M) of a wine. Sweetness and acidity are set independently by production choices.',
            'vars': {'X': 'sweetness', 'Z': 'acidity', 'M': 'flavor score'},
        },
        {
            'story': 'Both strong wind (X) and rough waves (Z) independently contribute to ships capsizing (M). Wind speed and wave height are uncorrelated at the event level.',
            'vars': {'X': 'wind strength', 'Z': 'wave roughness', 'M': 'capsizing events'},
        },
    ],
    'diamond': [
        {
            'story': 'A new CEO (X) changes both the marketing strategy (Y) and the R&D strategy (Z). Both marketing and R&D then influence the company stock price (W).',
            'vars': {'X': 'new CEO', 'Y': 'marketing strategy', 'Z': 'R&D strategy', 'W': 'stock price'},
        },
        {
            'story': 'A drug (X) affects both liver function (Y) and kidney function (Z). Both liver and kidney function influence overall health outcome (W).',
            'vars': {'X': 'drug dosage', 'Y': 'liver function', 'Z': 'kidney function', 'W': 'health outcome'},
        },
        {
            'story': 'Government policy (X) affects both education spending (Y) and infrastructure spending (Z). Both types of spending influence economic growth (W).',
            'vars': {'X': 'government policy', 'Y': 'education spending', 'Z': 'infrastructure spending', 'W': 'economic growth'},
        },
        {
            'story': 'A central bank rate change (X) affects both consumer spending (Y) and business investment (Z). Both consumer spending and business investment influence GDP growth (W).',
            'vars': {'X': 'central bank rate', 'Y': 'consumer spending', 'Z': 'business investment', 'W': 'GDP growth'},
        },
        {
            'story': 'An engine upgrade (X) improves both acceleration (Y) and top speed (Z). Both acceleration and top speed affect lap time (W).',
            'vars': {'X': 'engine upgrade', 'Y': 'acceleration', 'Z': 'top speed', 'W': 'lap time'},
        },
        {
            'story': 'A new curriculum (X) changes both science scores (Y) and humanities scores (Z). Both scores feed into overall GPA (W).',
            'vars': {'X': 'new curriculum', 'Y': 'science scores', 'Z': 'humanities scores', 'W': 'overall GPA'},
        },
        {
            'story': 'A climate policy (X) affects both industrial emissions (Y) and agricultural emissions (Z). Both emission sources feed into total CO2 (W).',
            'vars': {'X': 'climate policy', 'Y': 'industrial emissions', 'Z': 'agricultural emissions', 'W': 'total CO2'},
        },
    ],
}
