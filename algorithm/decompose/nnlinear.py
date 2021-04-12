import numpy as np
from scipy.optimize import nnls

class nnLinearSolver(object):
    def __init__(self):
        pass

    def insert_bcol(self, X):
        return np.insert(X, X.shape[1], 1, axis=1)

    def fit(self, X, y):
        W, _ = nnls(self.insert_bcol(X), y)
        self.weights = W[:-1]
        self.bias = W[-1]

    def predict(self, X):
        return list(map(lambda x: np.sum(x * self.weights) + self.bias, X))

# if __name__ == '__main__':
#     A = np.array([[1], [2]])
#     b = np.array([2,3])
#     model = nnLinearSolver()
#     model.train(A, b)
#     print(model.predict(A))