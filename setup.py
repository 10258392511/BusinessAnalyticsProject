from setuptools import setup, find_packages

setup(
    name="BusinessAnalyticsProject",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "xgboost",
        "scikit-optimize",
        "ipywidgets",
        "tqdm"
    ]
)
