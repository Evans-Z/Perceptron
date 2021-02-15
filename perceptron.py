import numpy as np

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0, scale=0.01, size=x.shape[1]+1)
        
        for i in range(self.n_iter):
            for x_i, y_i in zip(x, y):
                equal_or_not = np.where(self.forward(x_i)==y_i, 0, 1)
                self.w += equal_or_not * self.eta * y_i * np.append(1, x_i)
        
        return self

    
    def forward(self, x):
        x = x.reshape(-1, len(self.w) - 1)
        x0 = np.ones((x.shape[0], 1))
        predict = np.dot(self.w, np.append(x0, x, axis=1).T)
        return np.where(predict>=0, 1, -1)
        