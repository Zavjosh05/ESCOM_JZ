import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# =====================================
# 1️⃣ Cargar datos
# =====================================
data_x = pd.read_csv("datos.csv", usecols=[0])
data_y = pd.read_csv("datos.csv", usecols=[1])

print("Features (X):\n", data_x.T)
print("\nTargets (Y):\n", data_y.T)

# Dividir datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.3, train_size=0.7, shuffle=True, random_state=0
)

# Parámetros de SGD
max_iter = 10000  # número de iteraciones
alpha = 0.0001    # tasa de regularización

# Para guardar resultados
resultados = []

# =====================================
# 2️⃣ Función auxiliar para graficar
# =====================================
def graficar_modelo(x_test, y_test, y_pred, titulo, poly=None):
    plt.figure(figsize=(7, 4))
    plt.scatter(x_test, y_test, color="black", label="Datos reales")
    if poly is not None:
        # ordenar los puntos para una curva suave
        x_line = np.linspace(x_test.min(), x_test.max(), 200).reshape(-1, 1)
        y_line = model.predict(poly.transform(x_line))
        plt.plot(x_line, y_line, color="red", label="Predicción")
    else:
        plt.plot(x_test, y_pred, color="red", label="Predicción")
    plt.title(titulo)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

# =====================================
# 3️⃣ Regresiones con OLS
# =====================================

# --- Lineal OLS ---
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
resultados.append(["Regresión lineal con OLS", mse, r2])
graficar_modelo(x_test, y_test, y_pred, "Regresión lineal con OLS")

# --- Polinomial grado 2 OLS ---
poly2 = PolynomialFeatures(degree=2)
x_train_poly2 = poly2.fit_transform(x_train)
x_test_poly2 = poly2.transform(x_test)

model = LinearRegression()
model.fit(x_train_poly2, y_train)
y_pred = model.predict(x_test_poly2)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
resultados.append(["Regresión polinomial grado 2 con OLS", mse, r2])
graficar_modelo(x_test, y_test, y_pred, "Regresión polinomial grado 2 con OLS", poly2)

# --- Polinomial grado 3 OLS ---
poly3 = PolynomialFeatures(degree=3)
x_train_poly3 = poly3.fit_transform(x_train)
x_test_poly3 = poly3.transform(x_test)

model = LinearRegression()
model.fit(x_train_poly3, y_train)
y_pred = model.predict(x_test_poly3)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
resultados.append(["Regresión polinomial grado 3 con OLS", mse, r2])
graficar_modelo(x_test, y_test, y_pred, "Regresión polinomial grado 3 con OLS", poly3)

# =====================================
# 4️⃣ Regresiones con SGD
# =====================================

# Escaladores
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# --- Lineal SGD ---
X_train_scaled = scaler_x.fit_transform(x_train)
X_test_scaled = scaler_x.transform(x_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

sgd = SGDRegressor(max_iter=max_iter, alpha=alpha, tol=1e-3, random_state=0)
sgd.fit(X_train_scaled, y_train_scaled.ravel())
y_pred_scaled = sgd.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
resultados.append(["Regresión lineal con SGD", mse, r2])
graficar_modelo(x_test, y_test, y_pred, "Regresión lineal con SGD")

# --- Polinomial grado 2 SGD ---
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(x_train)
X_test_poly2 = poly2.transform(x_test)

scaler_x2 = StandardScaler()
X_train_poly2_scaled = scaler_x2.fit_transform(X_train_poly2)
X_test_poly2_scaled = scaler_x2.transform(X_test_poly2)

sgd2 = SGDRegressor(max_iter=max_iter, alpha=alpha, tol=1e-3, random_state=0)
sgd2.fit(X_train_poly2_scaled, y_train_scaled.ravel())
y_pred_scaled = sgd2.predict(X_test_poly2_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
resultados.append(["Regresión polinomial grado 2 con SGD", mse, r2])
graficar_modelo(x_test, y_test, y_pred, "Regresión polinomial grado 2 con SGD", poly2)

# --- Polinomial grado 3 SGD ---
poly3 = PolynomialFeatures(degree=3)
X_train_poly3 = poly3.fit_transform(x_train)
X_test_poly3 = poly3.transform(x_test)

scaler_x3 = StandardScaler()
X_train_poly3_scaled = scaler_x3.fit_transform(X_train_poly3)
X_test_poly3_scaled = scaler_x3.transform(X_test_poly3)

sgd3 = SGDRegressor(max_iter=max_iter, alpha=alpha, tol=1e-3, random_state=0)
sgd3.fit(X_train_poly3_scaled, y_train_scaled.ravel())
y_pred_scaled = sgd3.predict(X_test_poly3_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
resultados.append(["Regresión polinomial grado 3 con SGD", mse, r2])
graficar_modelo(x_test, y_test, y_pred, "Regresión polinomial grado 3 con SGD", poly3)

df_resultados = pd.DataFrame(resultados, columns=["Modelo", "MSE", "R2"])
print("\n" + "="*50)
print("RESUMEN DE RESULTADOS")
print("="*50)
print(df_resultados.to_string(index=False, formatters={
    "MSE": "{:.6f}".format,
    "R2": "{:.6f}".format
}))
