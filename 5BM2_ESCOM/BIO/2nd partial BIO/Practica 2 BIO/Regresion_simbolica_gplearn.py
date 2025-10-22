# ALUMNOS
# Díaz Alvarado Daniel Alejandro
# Zavaleta Guerrero Joshua Ivan

import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import graphviz

# -----------------------------
# Generación de datos sintéticos
# -----------------------------
x0 = np.arange(-1, 1, 1/10.)  # Rango de valores para x0
x1 = np.arange(-1, 1, 1/10.)  # Rango de valores para x1
x0_grid, x1_grid = np.meshgrid(x0, x1)  # Crea malla 2D
y_truth = x0_grid**2 - x1_grid**2 + x1_grid - 1  # Función real (verdad de terreno)

# -----------------------------
# Conjunto de entrenamiento y prueba
# -----------------------------
rng = np.random.RandomState(0)  # Fijar semilla para reproducibilidad
X = rng.uniform(-1, 1, 100).reshape(50, 2)  # 50 muestras de entrenamiento
y = X[:, 0]**2 - X[:, 1]**2 + X[:, 1] - 1   # Etiquetas reales para entrenamiento

X_test = rng.uniform(-1, 1, 100).reshape(50, 2)  # 50 muestras de prueba
y_test = X_test[:, 0]**2 - X_test[:, 1]**2 + X_test[:, 1] - 1  # Etiquetas reales de prueba

# -----------------------------
# Creación y entrenamiento del regresor simbólico
# -----------------------------
est_gp = SymbolicRegressor(
    population_size=5000,       # Tamaño de la población
    generations=20,             # Número de generaciones
    stopping_criteria=0.01,     # Criterio de parada por error mínimo
    p_crossover=0.7,            # Probabilidad de cruce
    p_subtree_mutation=0.1,     # Mutación de subárbol
    p_hoist_mutation=0.05,      # Mutación hoist
    p_point_mutation=0.1,       # Mutación puntual
    max_samples=0.9,            # Fracción de muestras usadas
    verbose=1,                  # Mostrar progreso
    parsimony_coefficient=0.01, # Penalización por complejidad
    random_state=0              # Semilla aleatoria
)

est_gp.fit(X, y)  # Entrenar el modelo simbólico
print("Programa simbólico encontrado:", est_gp._program)  # Mostrar expresión resultante

# -----------------------------
# Modelos de referencia (comparación)
# -----------------------------
est_tree = DecisionTreeRegressor()  # Árbol de decisión
est_tree.fit(X, y)
est_rf = RandomForestRegressor()    # Bosque aleatorio
est_rf.fit(X, y)

# -----------------------------
# Predicciones sobre la malla
# -----------------------------
y_gp = est_gp.predict(np.c_[x0_grid.ravel(), x1_grid.ravel()]).reshape(x0_grid.shape)
score_gp = est_gp.score(X_test, y_test)  # R² del modelo simbólico

y_tree = est_tree.predict(np.c_[x0_grid.ravel(), x1_grid.ravel()]).reshape(x0_grid.shape)
score_tree = est_tree.score(X_test, y_test)  # R² del árbol

y_rf = est_rf.predict(np.c_[x0_grid.ravel(), x1_grid.ravel()]).reshape(x0_grid.shape)
score_rf = est_rf.score(X_test, y_test)  # R² del random forest

# -----------------------------
# Visualización de resultados 3D
# -----------------------------
fig = plt.figure(figsize=(12, 10))
for i, (y_val, sc, title) in enumerate([
    (y_truth, None, "Ground Truth"),           # Función original
    (y_gp, score_gp, "SymbolicRegressor"),     # Modelo simbólico
    (y_tree, score_tree, "DecisionTreeRegressor"), # Árbol
    (y_rf, score_rf, "RandomForestRegressor")  # Bosque aleatorio
]):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')  # Subgráfica 3D
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.plot_surface(x0_grid, x1_grid, y_val, rstride=1, cstride=1, color='purple', alpha=0.5)
    ax.scatter(X[:, 0], X[:, 1], y, c='red')  # Puntos de entrenamiento
    if sc is not None:
        ax.text(-0.7, 1, 0.2, f"$R^2 = {sc:.6f}$", fontsize=14)  # Mostrar R²
    plt.title(title)
plt.show()

# -----------------------------
# Exportar el árbol simbólico final
# -----------------------------
try:
    dot_data = est_gp._program.export_graphviz()  # Exportar a formato Graphviz
    graph = graphviz.Source(dot_data)
    graph.render("best_program", format="png", cleanup=True)  # Guardar como imagen
    print("Árbol simbólico exportado como best_program.png")
except Exception as e:
    print("No se pudo exportar el árbol simbólico:", e)

# -----------------------------
# Visualizar información del proceso de crossover (si existe)
# -----------------------------
if hasattr(est_gp._program, "parents") and est_gp._program.parents is not None:
    parents = est_gp._program.parents
    print("Método de generación del mejor individuo:", parents.get("method", "Desconocido"))
    if "donor_idx" in parents and parents["donor_nodes"]:
        try:
            idx = parents["donor_idx"]  # Índice del donante
            fade_nodes = parents["donor_nodes"]  # Nodos utilizados del donante
            donor_prog = est_gp._programs[-2][idx]  # Programa donante
            dot_data = donor_prog.export_graphviz(fade_nodes=fade_nodes)
            donor_graph = graphviz.Source(dot_data)
            donor_graph.render("donor_program", format="png", cleanup=True)
            print("Árbol donante exportado como donor_program.png")
        except Exception as e:
            print("No se pudo visualizar el árbol donante:", e)
else:
    print("No hay información de los padres para este programa (posiblemente fue mutación o simplificación final).")
