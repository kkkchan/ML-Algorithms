import numpy as np
import tensorflow as tf
from tensorflow import keras

class GAT_conv(keras.layers.Layer):
    def __init__(self, units, activation='elu', use_bias=False, 
                 attention_heads=1, attention_reduction='concat', kernel_regularizer=None,
                 bias_regularizer=None, attn_kernel_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, attn_kernel_constraint=None,
                 dropout_rate=0.6, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                 **kwargs):
        self.units = units
        self.attention_heads = attention_heads
        self.attention_reduction = attention_reduction
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.activation = keras.activations.get(activation)
        
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = keras.regularizers.get(attn_kernel_regularizer)

        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attn_kernel_constraint = keras.constraints.get(attn_kernel_constraint)
        
        
        if attention_reduction == 'concat':
            # concat multi-heads
            self.output_dim = self.units * self.attention_heads
        else:
            # average multi-heads
            self.output_dim = self.units
        
        self.kernels = []
        self.bias = []
        self.attention_kernels = []
        
        super().__init__(**kwargs)
    
    
    def build(self, input_shape):
        for head in range(self.attention_heads):
            kernel = self.add_weight(shape=(input_shape[0][-1], self.units), 
                                     initializer=self.kernel_initializer,
                                     regularizer=self.activity_regularizer,
                                     name='kernel_head{}'.format(head))
            self.kernels.append(kernel)
            if self.use_bias:
                bias = self.add_weight(shape=(self.units, ),
                                      initializer=self.bias_initializer,
                                      regularizer=self.activity_regularizer,
                                      name='bias_head{}'.format(head))
                self.bias.append(bias)
            
            attention_kernel_self = self.add_weight(shape=(self.units, 1),
                                                   initializer=self.kernel_initializer,
                                                   regularizer=self.attn_kernel_regularizer,
                                                   constraint=self.attn_kernel_constraint,
                                                   name='attention_kernel_self_head{}'.format(head))
            attention_kernel_neighbors = self.add_weight(shape=(self.units, 1),
                                                         initializer=self.kernel_initializer,
                                                         regularizer=self.attn_kernel_regularizer,
                                                         constraint=self.attn_kernel_constraint,
                                                         name='attention_kernel_neighbors_head{}'.format(head))
            self.attention_kernels.append([attention_kernel_self, attention_kernel_neighbors])
            self.built = True
        
    def call(self, inputs, training=False):
        X, A = inputs # [nxd, nxn]
        outputs = []
        for head in range(self.attention_heads):
            kernel = self.kernels[head] # dxd'
            attention_kernel = self.attention_kernels[head] # [d'x1, d'x1]
            
            features = X @ kernel # nxd'
            attention_self = features @ attention_kernel[0] # nx1
            attention_neighbors = features @ attention_kernel[1] # nx1
            
            
            a_weight = attention_self + tf.transpose(attention_neighbors)
            a_weight = tf.nn.leaky_relu(a_weight, alpha=0.2)
            #  a_weight *= A # 掩码，去掉无连接关系的值
            a_weight += -10e9 * (1.0-A) # 处理softmax中的无连接位置的值
             
            a_weight = tf.nn.softmax(a_weight)
            
            dropout_attention = keras.layers.Dropout(self.dropout_rate)(a_weight, training=training) # nxn
            dropout_features = keras.layers.Dropout(self.dropout_rate)(features, training=training) # nxd'
            
            node_features = dropout_attention @ dropout_features # nxd'
            
            if self.use_bias:
                node_features += self.bias[head] # nxd'
            outputs.append(node_features)
            
        if self.attention_reduction == 'concat':
            output = tf.concat(outputs, axis=1)
        else:
            output = tf.reduce_mean(outputs, axis=0)
        return self.activation(output)
    
    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape
    
    
def gat(feature_dim, node_num, label_num, dropout_rate=0.0, att_heads=8, use_bias=True, l2_reg=tf.keras.regularizers.l2(5e-4/2)):
    X_input = keras.Input(shape=(feature_dim, ))
    A_input = keras.Input(shape=(node_num, ))
    
    dropout1 = keras.layers.Dropout(dropout_rate)(X_input)
    z = GAT_conv(units=16, activation='elu', dropout_rate=dropout_rate, kernel_regularizer=l2_reg,
                 attention_reduction='concat', 
                 attention_heads=att_heads)([dropout1, A_input])
    dropout2 = keras.layers.Dropout(dropout_rate)(z)
    z = GAT_conv(units=label_num, activation='softmax', kernel_regularizer=l2_reg,
                 attention_reduction='mean', dropout_rate=dropout_rate, 
                 attention_heads=1)([dropout2, A_input])
    model = keras.Model(inputs=[X_input, A_input], outputs=z, name='GAT_model')
    return model