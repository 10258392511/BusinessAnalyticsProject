import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def count_duplicates(df: pd.DataFrame):
    sizes = df.groupby(["Store", "Dept", "Date"]).size()

    return sizes
