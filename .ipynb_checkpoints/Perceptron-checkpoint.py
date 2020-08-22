import numpy as np


class Perceptron:

    def __init__(self, x, y, learning_rate=0.2):
        if len(x) != len(y):
            raise AttributeError('X,Y数量不一致')
        elif len(x) == 0:
            raise AttributeError('请输入数据')
        elif x.ndim > 2:
            raise AttributeError('张量X维度过多')

        self.X, self.Y = x, y
        self.w = np.ones((self.X.shape[1], 1), dtype=np.float32)
        self.b = 0
        self.l_rate = learning_rate
        self.iter_times = 0

    def fit(self, iteration_times=1000):
        is_finished = False
        while not is_finished and self.iter_times < iteration_times:
            wrong_count = 0
            self.iter_times += 1
            for d in range(len(self.X)):
                x = self.X[d]
                y = self.Y[d]
                if y * self.sign(x, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, x)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                is_finished = True
            if self.iter_times % 10 == 0:
                print('迭代次数：{}次; 误分类点：{}个'.format(self.iter_times, wrong_count))
        print('Fit complished!')

    @staticmethod
    def sign(x, w, b):
        y = np.dot(x, w) + b
        return y


if __name__ == '__main__':
    from sklearn import datasets
    # from sklearn.linear_model import Perceptron
    data = datasets.load_iris()
    x_train = data.data[:100, [0, 1]]
    y_train = np.array([1 if i > 0 else -1 for i in data.target[:100]])

    # model = Perceptron(fit_intercept=True, max_iter=10000, shuffle=False)
    model = Perceptron(x_train, y_train, 0.2)
    model.fit()

    import matplotlib.pyplot as plt
    x_ = np.linspace(4, 7, 10)
    # y_ = -(model.coef_[0][0] * x_ + model.intercept_) / model.coef_[0][1]
    y_ = -(model.w[0] * x_ + model.b) / model.w[1]
    plt.plot(x_, y_)
    plt.plot(x_train[:50, 0], x_train[:50, 1], 'bo', color='blue', label='negative')
    plt.plot(x_train[50:100, 0], x_train[50:100, 1], 'bo', color='orange', label='positive')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()
