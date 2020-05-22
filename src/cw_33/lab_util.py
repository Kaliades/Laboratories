import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit


def get_data_frame_from_csv(file_name):
    df = pd.read_csv(file_name)
    df = df.replace(to_replace=r'[,]', value='.', regex=True)
    df = df.astype(float)
    return df


def plot_all(df):
    sns.set(style="darkgrid")
    sns.lmplot(y="U[V] R=2cm", x="I[A] R=2cm", data=df, order=2,
               ci=None, scatter_kws={"s": 50}, height=10)
    sns.lmplot(y="U[V] R=3cm", x="I[A] R=3cm", data=df, order=2,
               ci=None, scatter_kws={"s": 50}, height=10)
    sns.lmplot(y="U[V] R=5cm", x="I[A] R=5cm", data=df, order=2,
               ci=None, scatter_kws={"s": 50}, height=10)


def fun(x, a):
    return a * np.power(x, 2)


def regression(data_frame):
    models = dict()
    r = 2
    for i in range(0, data_frame.columns.size - 1, 2):
        x = data_frame[data_frame.columns[i]].dropna().to_numpy()  # I[A]
        y = data_frame[data_frame.columns[i + 1]].dropna().to_numpy()
        popt, pcov = curve_fit(fun, x, y)  #
        model = np.poly1d([popt[0], 0, 0])
        if r == 4:
            r = 5
        models[r] = model
        r += 1
    return models
