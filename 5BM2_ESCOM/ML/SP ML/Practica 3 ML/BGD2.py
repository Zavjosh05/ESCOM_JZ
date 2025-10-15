import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def obtencion_de_w(x, y, paso=0.01, w=None):
    if w is None:
        w = [0] * len(x.columns)

    n = len(x)
    nw = len(w)

    for i in range(nw):
        calc = 0
        for j in range(n):
            y_pred_j = w[i] * x.iloc[j, i]
            calc += (y_pred_j - y.iloc[j, 0]) * x.iloc[j, i]

        w[i] -= float(2 * paso * calc)
    return w


def bgd_artesanal(x_train, y_train, x_test, y_test, paso=0.01, w=None):
    wn = obtencion_de_w(x_train, y_train, paso=paso, w=w)

    nw = len(wn)
    n = len(x_test)

    y_pred = [0] * n
    error_estimacion = 0
    for i in range(n):
        y_pred[i] = float(sum(x_test.iloc[i, j] * wn[j] for j in range(nw)))
        error_estimacion += abs(y_pred[i] - float(y_test.iloc[i, 0]))

    return y_pred, error_estimacion, wn


def bgd_artesanal_grafico(x_train, y_train, x_test, y_test, paso=0.01, w=None, iteraciones=1):
    y_pred_list = []
    error_list = []
    w_list = []

    if w is None:
        w = [0] * len(x_train.columns)

    for i in range(iteraciones):
        y_pred, error_estimacion, w = bgd_artesanal(x_train, y_train, x_test, y_test, paso=paso, w=w)
        y_pred_list.append(y_pred)
        error_list.append(error_estimacion)
        w_list.append(w.copy())

    print("Valores de w por iteración:")
    for i, wi in enumerate(w_list):
        print(f"Iteración {i}: {wi}")

    print("\nValores de y reales:\n", y_test.T)
    print("\nValores de y_pred:")
    for i, ypi in enumerate(y_pred_list):
        print(f"Iteración {i}: {ypi}")

    print("\nErrores de estimación:")
    for i, err in enumerate(error_list):
        print(f"Iteración {i}: {err}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, iteraciones + 1), error_list, marker='o', color='blue')
    plt.title("Error de estimación por iteración")
    plt.xlabel("Iteración")
    plt.ylabel("Error total")
    plt.grid(True)
    plt.show()

    return y_pred_list, error_list, w_list


data_x = pd.read_csv("Dataset_multivariable.csv", usecols=range(0,5))
data_y = pd.read_csv("Dataset_multivariable.csv", usecols=[5])

print("Features:\n", data_x.T, "\n\ntargets:\n", data_y.T)

x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.3, train_size=0.7, shuffle=True, random_state=0
)
x_train, x_test, y_train, y_test = (
    x_train.astype(float),
    x_test.astype(float),
    y_train.astype(float),
    y_test.astype(float),
)

print("\n\nTraining set\nFeatures:\n", x_train.T, "\n\nTargets:\n", y_train.T)
print("\nTest set\nFeatures:\n", x_test.T, "\n\nTargets:\n", y_test.T, "\n\nProceso para 4 iteraciones")

bgd_artesanal_grafico(x_train, y_train, x_test, y_test, paso=6e-6, iteraciones=4)
