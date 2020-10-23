import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self):
        self.model = None

    @staticmethod
    def mean(x):
        return sum(x) / float(len(x))

    def std(self, x):
        avg = self.mean(x)
        return np.sqrt(sum([np.power(e-avg, 2) for e in x]) / float(len(x)))

    def gaussain_prob(self, x, mean, std):
        exp = np.exp(-(np.power(x - mean, 2) /
                       (2 * np.power(std, 2))))
        return (1 / np.sqrt(2 * np.pi) * std(x)) * exp

    def fit(self, x_train, y_train):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(x, y):
            data[label].append(f)


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    x = iris.data[:100]
    y = iris.target[:100]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
