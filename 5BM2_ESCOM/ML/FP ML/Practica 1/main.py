import pandas as pd
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.utils import resample

data = pd.read_csv("metodosDeValidacion.csv")
# dataX = pd.read_csv("metodosDeValidacion.csv", usecols= ["x"])
# dataY = pd.read_csv("metodosDeValidacion.csv", usecols= ["y"])

print("Features(x) and target(y):\n",data.T,"\n")

x_train, x_test, y_train, y_test = train_test_split(data,data, test_size=0.3, train_size=0.7, shuffle=False)

print("Conjunto de entrenamiento:\n",x_train.T)
print("\nConjunto de prueba:\n",x_test.T)

print("\nKFolds")
kf = KFold(n_splits=6)

for i, (train_index, test_index) in enumerate(kf.split(x_train)):
    k_train = x_train.iloc[train_index]
    k_test = x_train.iloc[test_index]

    print("Pliegue: ", i+1)
    print("Conjunto de entrenamiento: \n", k_train.T,"\n")
    print("Conjunto de prueba: \n", k_test.T,"\n")

print("\nLeave one out")
LOO = LeaveOneOut()
LOO.get_n_splits(x_train)

for i, (train_index, test_index) in enumerate(LOO.split(x_train)):
    k_train = x_train.iloc[train_index]
    k_test = x_train.iloc[test_index]
    print("Pliegue: ", i + 1)
    print("Conjunto de entrenamiento: \n", k_train.T, "\n")
    print("Conjunto de prueba: \n", k_test.T, "\n")

print("\nBootstrap\n")

for i in range(1,3):
    features, targets_nu =resample(x_train,x_train,replace=True,n_samples=9,)

    targets = x_train.loc[~x_train.index.isin(features.index)]

    print(f"C{i}\n")
    print("Conjunto de entrenamiento: \n", features.T,"\n")
    print("Conjunto de prueba: \n", targets.T,"\n")

