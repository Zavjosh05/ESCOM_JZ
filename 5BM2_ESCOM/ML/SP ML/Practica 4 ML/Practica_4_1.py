import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

data_x = pd.read_csv("datos.csv", usecols=[0])
data_y = pd.read_csv("datos.csv", usecols=[1])

x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.3, train_size=0.7, shuffle=True, random_state=0
)

max_iter = 10000
alpha = 0.0000001
resultados = []

def graficar_modelo(x_test, y_test, model, titulo, poly=None, scaler_x=None):
    plt.figure(figsize=(7, 4))
    plt.scatter(x_test, y_test, color="black", label="Datos reales")

    x_line = np.linspace(x_test.min(), x_test.max(), 200).reshape(-1, 1)
    if poly is not None:
        x_line_t = poly.transform(x_line)
    else:
        x_line_t = x_line

    # Sin escalador: pasamos directamente x_line_t al modelo
    y_line = model.predict(x_line_t)
    plt.plot(x_line, y_line, color="red", label="Predicción")

    plt.title(titulo)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

#OLS
for grado in [1, 2, 3]:
    if grado == 1:
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        titulo = "Regresión lineal con OLS"
        poly = None
    else:
        poly = PolynomialFeatures(degree=grado)
        x_train_poly = poly.fit_transform(x_train)
        x_test_poly = poly.transform(x_test)
        model = LinearRegression()
        model.fit(x_train_poly, y_train)
        y_pred = model.predict(x_test_poly)
        titulo = f"Regresión polinomial grado {grado} con OLS"

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    resultados.append([titulo, mse, r2])
    graficar_modelo(x_test, y_test, model, titulo, poly)

#sgd
for grado in [1, 2, 3]:
    if grado == 1:
        X_train_s = x_train.values
        X_test_s = x_test.values
        poly = None
        titulo = "Regresión lineal con SGD"
    else:
        poly = PolynomialFeatures(degree=grado)
        X_train_s = poly.fit_transform(x_train)
        X_test_s = poly.transform(x_test)
        titulo = f"Regresión polinomial grado {grado} con SGD"

    sgd = SGDRegressor(max_iter=max_iter, eta0=alpha, learning_rate='constant', random_state=0)
    sgd.fit(X_train_s, y_train.values.ravel())

    y_pred = sgd.predict(X_test_s)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    resultados.append([titulo, mse, r2])

    graficar_modelo(x_test, y_test, sgd, titulo, poly)

#resultados
df_resultados = pd.DataFrame(resultados, columns=["Modelo", "MSE", "R2"])
print("\n" + "="*60)
print("RESUMEN DE RESULTADOS")
print("="*60)
print(df_resultados.to_string(index=False, formatters={
    "MSE": "{:.6f}".format,
    "R2": "{:.6f}".format
}))
