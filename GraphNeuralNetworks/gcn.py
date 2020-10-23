import tensorflow as tf
from tensorflow.keras import layers
from dgl.nn import GraphConv

class GCN(tf.keras.Model):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layer_list = []
        # input layer
        self.layer_list.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layer_list.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layer_list.append(GraphConv(n_hidden, n_classes))
        self.dropout = layers.Dropout(dropout)

    def call(self, features):
        h = features
        for i, layer in enumerate(self.layer_list):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h