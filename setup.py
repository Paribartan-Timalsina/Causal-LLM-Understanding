from setuptools import find_packages, setup

setup(
    name='causal-llm-eval',
    version='1.0.0',
    description="Probing causal reasoning in LLMs across Pearl's hierarchy",
    author='Paribartan Timalsina',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.35.0',
        'accelerate>=0.24.0',
        'bitsandbytes>=0.41.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'networkx>=3.0',
        'tqdm>=4.65.0',
        'pandas>=2.0.0',
    ],
    entry_points={
        'console_scripts': [
            'causal-llm-eval=scripts.run:main',
        ],
    },
)
