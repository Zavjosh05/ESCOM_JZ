# ALUMNOS
# Díaz Alvarado Daniel Alejandro
# Zavaleta Guerrero Joshua Ivan

import math
import operator
import random
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from deap import gp, creator, base, tools, algorithms

# ===============================================================
# ============= CONFIGURACIÓN DE FUNCIONES BÁSICAS ==============
# ===============================================================

# Función de división protegida (evita divisiones por cero)
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# Potencias predefinidas para facilitar la evolución
def pow2(x):
    return x ** 2

def pow3(x):
    return x ** 3

# ===============================================================
# ============= DEFINICIÓN DEL CONJUNTO DE PRIMITIVAS ===========
# ===============================================================

# El conjunto de primitivas define qué operaciones puede usar el algoritmo genético
# En este caso, funciones que operan sobre 2 variables (x, y)
pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(pow2, 1)
pset.addPrimitive(pow3, 1)

# Renombramos los argumentos para mayor claridad
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')

# ===============================================================
# ============= DEFINICIÓN DE LOS TIPOS GENÉTICOS ===============
# ===============================================================

# FitnessMin → queremos minimizar el error
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Individual → cada individuo es un árbol de primitivas
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# ===============================================================
# ============= CONFIGURACIÓN DE LA CAJA DE HERRAMIENTAS ========
# ===============================================================

toolbox = base.Toolbox()

# Cómo generar expresiones (árboles)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)

# Cómo generar individuos y poblaciones
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Cómo "compilar" un árbol de DEAP a una función de Python
toolbox.register("compile", gp.compile, pset=pset)

# ===============================================================
# ============= FUNCIÓN DE EVALUACIÓN DEL FITNESS ===============
# ===============================================================

# Función objetivo real:
# f(x, y) = 5*x^3*y^2 + x/2
def evalSymbReg(individual, points):
    func = toolbox.compile(expr=individual)
    sqerrors = []
    for (x, y) in points:
        try:
            val = func(x, y)
            real = 5*x**3*y**2 + x/2
            sqerrors.append((val - real) ** 2)
        except Exception:
            sqerrors.append(1e6)
    return (sum(sqerrors) / len(points),)

# Registramos la función de evaluación y los operadores genéticos
toolbox.register("evaluate", evalSymbReg,
                 points=[(x / 10.0, y / 10.0) for x in range(-10, 10) for y in range(-10, 10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# ===============================================================
# ============= ESTADÍSTICAS Y EVOLUCIÓN ========================
# ===============================================================

hof = tools.HallOfFame(10)  # Mejores 10 individuos
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

random.seed(318)

# Generamos población inicial y ejecutamos la evolución
pop = toolbox.population(n=200)
pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats=mstats, halloffame=hof, verbose=True)

# ===============================================================
# ============= FUNCIONES DE ANÁLISIS SIMBÓLICO =================
# ===============================================================

x, y = sp.symbols('x y')
target_expr = 5*x**3*y**2 + x/2  # Función objetivo simbólica

# Conversión del árbol de DEAP → expresión Sympy
def deap_to_sympy(individual):
    expr_str = str(individual)
    replacements = {
        "add": "sp.Add",
        "sub": "lambda a,b: a - b",
        "mul": "sp.Mul",
        "neg": "lambda a: -a",
        "protectedDiv": "lambda a,b: a/b if b != 0 else 1",
        "pow2": "lambda a: a**2",
        "pow3": "lambda a: a**3",
        "cos": "sp.cos",
        "sin": "sp.sin"
    }
    local_env = {"sp": sp, "x": sp.Symbol("x"), "y": sp.Symbol("y")}
    local_env.update({k: eval(v, {"sp": sp}) for k, v in replacements.items()})
    try:
        expr = eval(expr_str, {}, local_env)
        return sp.simplify(expr)
    except Exception as e:
        raise ValueError(f"Error al convertir '{expr_str}' → Sympy: {e}")

# -------- Similitud simbólica original (exacta) --------
def similitud_simbolica(expr_simplified, target_expr):
    expr_simplified = sp.simplify(expr_simplified)
    target_expr = sp.simplify(target_expr)
    terms_simplified = set(sp.Add.make_args(expr_simplified))
    terms_target = set(sp.Add.make_args(target_expr))
    comunes = terms_simplified.intersection(terms_target)
    similitud = len(comunes) / len(terms_target) * 100 if terms_target else 0
    return similitud

# -------- Similitud numérica (basada en correlación) --------
def similitud_numerica(expr1, expr2, n=25):
    f1 = sp.lambdify((x, y), expr1, "numpy")
    f2 = sp.lambdify((x, y), expr2, "numpy")
    X = np.linspace(-2, 2, n)
    Y = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(X, Y)
    Z1 = f1(X, Y)
    Z2 = f2(X, Y)
    mask = np.isfinite(Z1) & np.isfinite(Z2)
    if np.sum(mask) == 0:
        return 0
    corr = np.corrcoef(Z1[mask].flatten(), Z2[mask].flatten())[0, 1]
    return float(max(0, corr) * 100)  # Aseguramos que no haya negativos

# ===============================================================
# ============= RESULTADOS Y VISUALIZACIÓN ======================
# ===============================================================

print("\n===== HALL OF FAME (primeros 10 individuos) =====\n")
for i, ind in enumerate(hof[:10]):
    try:
        simplified = deap_to_sympy(ind)
        diff_expr = sp.simplify(simplified - target_expr)
        sim_sym = similitud_simbolica(simplified, target_expr)
        sim_num = similitud_numerica(simplified, target_expr)
    except Exception as e:
        simplified = f"Error al simplificar: {e}"
        diff_expr = "N/A"
        sim_sym = 0.0
        sim_num = 0.0

    print(f"Individuo {i+1}:")
    print(f" ➤ Expresión DEAP: {ind}")
    print(f" ➤ Simplificada: {simplified}")
    print(f" ➤ Diferencia simbólica: {diff_expr}")
    print(f" ➤ Similitud simbólica exacta: {sim_sym:.2f}%")
    print(f" ➤ Similitud numérica: {sim_num:.2f}%")
    print(f" ➤ Fitness: {ind.fitness.values[0]:.6f}")
    print("-" * 80)

# ===============================================================
# ============= GRAFICAR MEJOR INDIVIDUO ========================
# ===============================================================

# Obtenemos el mejor individuo (el primero del Hall of Fame)
best_ind = hof[0]
best_index = 1  # Siempre el primero del Hall of Fame

print("\n===== MEJOR INDIVIDUO ENCONTRADO =====")
print(f"🏆 Individuo #{best_index} del Hall of Fame")
print(f"Expresión DEAP: {best_ind}")

# Convertimos el individuo a expresión simbólica
best_expr = deap_to_sympy(best_ind)
print(f"Expresión simbólica simplificada: {best_expr}")

# Calculamos su similitud numérica respecto a la función objetivo
sim_num_best = similitud_numerica(best_expr, target_expr)
print(f"🔹 Similitud numérica con la función real: {sim_num_best:.2f}%")
print(f"🔹 Fitness del individuo: {best_ind.fitness.values[0]:.6f}")

# ===============================================================
# ============= GRAFICAR FUNCIÓN REAL Y EVOLUCIONADA ============
# ===============================================================

f_real = sp.lambdify((x, y), target_expr, "numpy")
f_best = sp.lambdify((x, y), best_expr, "numpy")

# Malla para graficar
X = np.linspace(-2, 2, 60)
Y = np.linspace(-2, 2, 60)
X, Y = np.meshgrid(X, Y)

# Evaluamos ambas funciones y su error absoluto
Z_real = f_real(X, Y)
Z_best = f_best(X, Y)
Z_error = np.abs(Z_real - Z_best)

# ========= Gráficas =========
fig = plt.figure(figsize=(18, 5))

# Función objetivo
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z_real, alpha=0.85)
ax1.set_title("Función objetivo: 5x³y² + x/2")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Mejor individuo encontrado
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z_best, alpha=0.85)
ax2.set_title("Mejor individuo evolutivo")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Error absoluto entre ambas
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z_error, alpha=0.85)
ax3.set_title("|Error| entre ambas funciones")
ax3.set_xlabel("x")
ax3.set_ylabel("y")

plt.tight_layout()
plt.show()
