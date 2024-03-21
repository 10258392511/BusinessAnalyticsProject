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
        "plotly",
        "nbformat",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "eli5",
        "scikit-optimize",
        "ipykernel",
        "ipywidgets",
        "pyarrow",
        "PyYAML",
        "tqdm"
    ]
)
