from Multi_layer_NNs import NN
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    data = datasets.load_iris()
    x = data.data[:100, [0,1]]
    y = np.array([1 if i > 0 else 0 for i in data.target[:100]])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = NN([2, 3, 1])
    model.fit(x_train, y_train.reshape(len(y_train), -1))
    print('Train Score:{}'.format(model.score(x_train, y_train)))
    print('Test Score:{}'.format(model.score(x_test, y_test)))

    plt.subplot(211)
    # x_samples = np.linspace(4, 7, 500)
    # y_samples = (- model.b - model.w[0]*x_samples) / model.w[1]
    # plt.plot(x_samples, y_samples, 'r')
    plt.scatter(x[:50, 0], x[:50, 1], label='negative')
    plt.scatter(x[50:100, 0], x[50:100, 1], label='positive')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title('LosRes Results on iris datasets')
    plt.legend()
    
    plt.subplots_adjust(hspace=0.5,wspace=0.25)
    plt.subplot(212)
    plt.plot(range(len(model.loss)), model.loss, '-o')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title('loss function')
    plt.show()