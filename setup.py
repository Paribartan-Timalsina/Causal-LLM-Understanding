from setuptools import setup, find_packages

setup(
    name="causal-llm-eval",
    version="1.0.0",
    description="Probing causal inference capabilities in LLMs across Pearl's Hierarchy",
    author="Paribartan Timalsina",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "networkx>=3.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "causal-llm-eval=scripts.run_evaluation:main",
            "causal-llm-analysis=scripts.run_analysis:main",
            "causal-llm-report=scripts.generate_report:main",
        ],
    },
)
