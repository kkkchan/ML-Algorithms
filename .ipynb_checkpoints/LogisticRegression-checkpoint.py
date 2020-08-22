import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, max_iter=1000, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.loss_ = []
        print('LogisticRegression Model(learning_rate={}, max_iteration={})'.format(
            self.learning_rate, self.max_iter))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        self.w = np.zeros((x.shape[1], 1), dtype=np.float32)
        iter_times = 0
        while iter_times < self.max_iter:
            for i in range(len(x)):
                res = self.sigmoid(np.dot(x[i], self.w))
                self.w += self.learning_rate * (y[i] - res) * x[i].reshape(3,1)
            iter_times += 1
            if iter_times % 50 == 0:
                loss_temp = self.loss(x, y)
                self.loss_.append(loss_temp)
                print('迭代次数:{}次; 损失函数:{}'.format(iter_times, loss_temp))
        print('Fit complished!')

    def loss(self, x, y):
        return -1 * np.sum(y.reshape((len(y), 1)) * np.log(self.sigmoid(np.dot(x, self.w))) +
                    (1 - y.reshape((len(y), 1))) * np.log(1 - self.sigmoid(np.dot(x, self.w)))) / len(x)

    def predict(self, x):
        return self.sigmoid(np.dot(x, self.w))

    def score(self, x_test, y_test):
        x_pre = self.sigmoid(np.dot(x_test, self.w))

        y_pre = np.array([1 if i > 0.5 else 0 for i in x_pre])
        score = np.sum(y_pre == y_test) / len(y_pre)
        print(score)

if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    data = datasets.load_iris()
    x = data.data[:100, [0,1]]
    x = np.hstack([np.ones((100, 1)), x])
    y = np.array([1 if i > 0 else 0 for i in data.target[:100]])

    x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size=0.3)


    model = LogisticRegression()
    model.fit(x_train, y_train)
    x_points = np.arange(4, 8)
    y_ = -(model.w[1]*x_points + model.w[0]) / model.w[2]
    print('Score:{}'.format(model.score(x_test, y_test)))

    plt.plot(x_points, y_)
    plt.scatter(x[:50, 1], x[:50, 2], label='negative')
    plt.scatter(x[50:100, 1], x[50:100, 2], label='positive')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title('LosRes Results on iris datasets')
    plt.legend()
    plt.show()

    plt.plot(range(len(model.loss_)), model.loss_, '-o')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title('loss function')
    plt.show()