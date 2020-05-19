from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import re
from sympy import *
import pandas as pd

__all__ = ['plot_model', 'create_plot_for_poly', 'find_u_max_and_i_max', 'parse_data', 'calculate_and_print_results',
           'print_model']


def plot_model(x, y, model, line):
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y)
    plt.plot(line, model(line))
    plt.show()


def create_plot_for_poly(x, y, line):
    plt.figure(figsize=(15, 15))
    plt.scatter(x, y)
    legend = []
    colors = ['#272727', '#D90368', '#ACADBC', '#5F5AA2', '#88D188']
    i = 0
    for n in range(2, 7):
        model = np.poly1d(np.polyfit(x, y, n))
        y_pred = model(x)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        title = f'Wielomian {n} R2:{r2} RMSE:{rmse}'
        legend_element, = plt.plot(line, model(line), label=title, linewidth=3, c=colors[i])
        legend.append(legend_element)
        i += 1

    plt.legend(handles=legend)
    plt.ylabel('I[mA]')
    plt.xlabel('U[V]')
    plt.title('Regresja wielomianowa')
    plt.show()


def parse_data(form_file, to_file):
    with open(form_file, newline='') as file:
        text = file.read()
        text = re.sub("[,]", ".", text)
        text = re.sub("[ ]", ",", text)
    with open(to_file, mode='w') as file:
        file.write(text)


def find_u_max_and_i_max(model, sesitive=0.1):
    result = dict()
    for x in np.arange(0, 1.6, sesitive):
        y_pred = model(x)
        result[(y_pred * x)] = x
    u_max = max(result.items())[1]
    i_max = model(u_max)
    return [u_max, i_max]


def calculate_and_print_results(model):
    u_max, i_max = find_u_max_and_i_max(model)
    i_sc = model(0)
    u_oc = np.real(model.roots[0])
    p_max = u_max * i_max
    FF = p_max / (u_oc * i_sc)
    p_light = 11.87
    eta = p_max / p_light * 100
    df = pd.DataFrame(data={
        'U_max [V]': [u_max], 'I_max [mA]': [i_max], 'I_sc [mA]': [i_sc], 'U_oc [V]': [u_oc], "FF": [FF],
        'P_max [W]': p_max, "\u03B7 [%]": eta
    })
    return df


def print_model(model):
    x = symbols('x')
    return expand(model(x))
