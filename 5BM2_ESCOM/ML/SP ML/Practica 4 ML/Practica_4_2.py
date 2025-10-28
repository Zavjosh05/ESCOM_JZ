import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

# =====================================
# 1️⃣ Cargar datos
# =====================================
data = pd.read_csv("cal_housing.csv")

# Última columna = target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# División 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, shuffle=True, random_state=0
)

resultados = []

# =====================================
# 2️⃣ Función auxiliar para evaluar modelo
# =====================================
def evaluar_modelo(model, X_test, y_test, nombre):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    resultados.append([nombre, mse, r2])

# =====================================
# 3️⃣ Modelos con OLS
# =====================================

# --- Lineal OLS ---
model = LinearRegression()
model.fit(X_train, y_train)
evaluar_modelo(model, X_test, y_test, "Regresión lineal con OLS")

# --- Polinomial grado 2 ---
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly2, y_train)
evaluar_modelo(model, X_test_poly2, y_test, "Regresión polinomial grado 2 con OLS")

# --- Polinomial grado 2 con StandardScaler ---
scaler_std = StandardScaler()
X_train_poly2_std = scaler_std.fit_transform(X_train_poly2)
X_test_poly2_std = scaler_std.transform(X_test_poly2)
model = LinearRegression()
model.fit(X_train_poly2_std, y_train)
evaluar_modelo(model, X_test_poly2_std, y_test, "Regresión polinomial grado 2 con StandardScaler")

# --- Polinomial grado 2 con RobustScaler ---
scaler_rob = RobustScaler()
X_train_poly2_rob = scaler_rob.fit_transform(X_train_poly2)
X_test_poly2_rob = scaler_rob.transform(X_test_poly2)
model = LinearRegression()
model.fit(X_train_poly2_rob, y_train)
evaluar_modelo(model, X_test_poly2_rob, y_test, "Regresión polinomial grado 2 con RobustScaler")

# --- Polinomial grado 3 ---
poly3 = PolynomialFeatures(degree=3)
X_train_poly3 = poly3.fit_transform(X_train)
X_test_poly3 = poly3.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly3, y_train)
evaluar_modelo(model, X_test_poly3, y_test, "Regresión polinomial grado 3 con OLS")

# --- Polinomial grado 3 con StandardScaler ---
scaler_std3 = StandardScaler()
X_train_poly3_std = scaler_std3.fit_transform(X_train_poly3)
X_test_poly3_std = scaler_std3.transform(X_test_poly3)
model = LinearRegression()
model.fit(X_train_poly3_std, y_train)
evaluar_modelo(model, X_test_poly3_std, y_test, "Regresión polinomial grado 3 con StandardScaler")

# --- Polinomial grado 3 con RobustScaler ---
scaler_rob3 = RobustScaler()
X_train_poly3_rob = scaler_rob3.fit_transform(X_train_poly3)
X_test_poly3_rob = scaler_rob3.transform(X_test_poly3)
model = LinearRegression()
model.fit(X_train_poly3_rob, y_train)
evaluar_modelo(model, X_test_poly3_rob, y_test, "Regresión polinomial grado 3 con RobustScaler")

# =====================================
# 4️⃣ Resumen
# =====================================
df_resultados = pd.DataFrame(resultados, columns=["Modelo", "MSE", "R2"])
print("\n" + "="*70)
print("RESUMEN DE RESULTADOS")
print("="*70)
print(df_resultados.to_string(index=False, formatters={
    "MSE": "{:.6f}".format,
    "R2": "{:.6f}".format
}))
