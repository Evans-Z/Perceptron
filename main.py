import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from perceptron import Perceptron
from tools import plot_decision_regions

df = pd.read_csv("./Iris.csv")

y = df.iloc[0:100, 4]
y = np.where(y == "setosa", -1, 1)
x = df.iloc[0:100, [0, 2]].values

my_perceptron = Perceptron(random_state=1)
par = my_perceptron.fit(x, y)
print(par.w)

plot_decision_regions(x, y, my_perceptron)