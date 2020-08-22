import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



class LogisticRegression:
    '''LogisticRegression for binary classification
    
    max_iter: the maximum iteration times for training
    learning_rate: learing rate for gradiend decsend training

    Input's shape should be [sample_nums, data_dims]

    attrs:
        max_iter
        learning_rate
        (after fit)
        w
        b
        costs

    methods:
        fit
        predict
        predict_proba
        score    
    '''

    def __init__(self, max_iter=2000, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        print('LogisticRegression Model(learning_rate={}, max_iteration={})'.format(
            self.learning_rate, self.max_iter))


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def initialize_with_zeros(self, dim):
        w = np.zeros((dim, 1))
        b = 0

        assert (w.shape == (dim, 1))
        assert (isinstance(b, float) or isinstance(b, int))

        return w, b    


    def propagate(self, w, b, X, Y):
        m = X.shape[0]
        A = self.sigmoid(np.dot(X, w) + b)
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

        dw = 1 / m * np.dot(X.T, A - Y)
        db = 1 / m * np.sum(A - Y) 

        assert (dw.shape == w.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)
        assert (cost.shape == ())
        grads = {'dw': dw,
                'db': db}

        return grads, cost


    def optimize(self, w, b, X, Y, max_iter, learning_rate, print_cost=False):
        costs = []
        for i in range(1, max_iter+1):
            grads, cost = self.propagate(w, b, X, Y)
            w -= learning_rate * grads['dw']
            b -= learning_rate * grads['db']

            if i % 100 == 0:
                costs.append(cost)
                if print_cost:
                    print('Cost after iteration %i: %f'%(i, cost))
        return w, b, costs


    def fit(self, X, Y, print_cost=False):
        print('Fit starting:')
        w, b = self.initialize_with_zeros(X.shape[1])
        iter_time = 0

        self.w, self.b, self.costs = self.optimize(w, b, X, Y, self.max_iter, self.learning_rate, print_cost)
        print('Fit complished!')


    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)


    def predict(self, X):
        proba = self.predict_proba(X)
        pre = np.zeros_like(proba, dtype=np.int)
        pre[proba > 0.5] = 1
        pre = np.squeeze(pre)
        return pre


    def score(self, X_test, Y_test):
        Y_pre = self.predict(X_test)
        score = np.sum(Y_pre == Y_test) / len(Y_pre)
        return score


    def __str__(self):
        return 'LogisticRegression Model(learning_rate={}, max_iteration={})'.format(
            self.learning_rate, self.max_iter)


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    data = datasets.load_iris()
    x = data.data[:100, [0,1]]
    # x = np.hstack([np.ones((100, 1)), x])
    y = np.array([1 if i > 0 else 0 for i in data.target[:100]])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = LogisticRegression()
    model.fit(x_train, y_train.reshape(len(y_train), -1), True)
    print('Train Score:{}'.format(model.score(x_train, y_train)))
    print('Test Score:{}'.format(model.score(x_test, y_test)))

    plt.subplot(211)
    x_samples = np.linspace(4, 7, 500)
    y_samples = (- model.b - model.w[0]*x_samples) / model.w[1]
    plt.plot(x_samples, y_samples, 'r')
    plt.scatter(x[:50, 0], x[:50, 1], label='negative')
    plt.scatter(x[50:100, 0], x[50:100, 1], label='positive')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title('LosRes Results on iris datasets')
    plt.legend()
    
    plt.subplots_adjust(hspace=0.5,wspace=0.25)
    plt.subplot(212)
    plt.plot(range(len(model.costs)), model.costs, '-o')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title('loss function')
    plt.show()