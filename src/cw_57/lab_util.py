from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


def create_plot_for_poly(x, y, line):
    scoreR2 = dict()
    scoreRMSE = dict()
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
        scoreR2[r2] = n
        scoreRMSE[rmse] = n
        title = f'Wielomain {n}'
        legend_element, = plt.plot(line, model(line), label=title, linewidth=3, c=colors[i])
        legend.append(legend_element)
        i += 1

    plt.legend(handles=legend)
    plt.show()
    minR2 = max(scoreR2.items())
    minRMSE = min(scoreRMSE.items())
    print(f'R2 = {minR2}')
    print(f'RMSE = {minRMSE}')
    scoreR2
