import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(x, y, classfier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'green', 'yellow', 'black')

    classes_number = len(np.unique(y))
    color_map = ListedColormap(colors[:classes_number])

    x1_min, x1_max = x[:,0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:,1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    z = classfier.forward(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, z, alpha=0.2, cmap=color_map)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    for idx, classes in enumerate(np.unique(y)):
        plt.scatter(x=x[y==classes, 0], y=x[y==classes, 1], color=colors[idx], marker=markers[idx])
    plt.show()