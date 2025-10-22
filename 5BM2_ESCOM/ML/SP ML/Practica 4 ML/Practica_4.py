import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_x = pd.read_csv("datos.csv", usecols=[0])
data_y = pd.read_csv("datos.csv", usecols=[1])

print("Features: ", data_x.T, "\n\ntargets:", data_y.T, "\n")

x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.3, train_size=0.7, shuffle=True, random_state=0
)

