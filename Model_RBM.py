import tensorflow as tf
import numpy as np
from keras import backend as K


class RBM(tf.keras.Model):
    def __init__(self, num_visible, num_hidden):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        # RBM parameters
        self.W = tf.Variable(tf.random.normal([num_visible, num_hidden], mean=0.0, stddev=0.01))
        self.visible_bias = tf.Variable(tf.zeros([num_visible]))
        self.hidden_bias = tf.Variable(tf.zeros([num_hidden]))

    def sample_hidden(self, visible_probabilities):
        hidden_probabilities = tf.sigmoid(tf.matmul(visible_probabilities, self.W) + self.hidden_bias)
        return hidden_probabilities, tf.keras.backend.random_binomial(tf.shape(hidden_probabilities),
                                                                      p=hidden_probabilities)

    def sample_visible(self, hidden_probabilities):
        visible_probabilities = tf.sigmoid(tf.matmul(hidden_probabilities, tf.transpose(self.W)) + self.visible_bias)
        return visible_probabilities, tf.keras.backend.random_binomial(tf.shape(visible_probabilities),
                                                                       p=visible_probabilities)

    def gibbs_sampling(self, visible_probabilities):
        hidden_probabilities, sampled_hidden = self.sample_hidden(visible_probabilities)
        visible_probabilities, sampled_visible = self.sample_visible(hidden_probabilities)
        return visible_probabilities, sampled_visible, hidden_probabilities, sampled_hidden

    def call(self, inputs):
        # Perform Gibbs sampling for training
        v0 = inputs
        h0_prob, h0 = self.sample_hidden(v0)
        v1_prob, v1, h1_prob, h1 = self.gibbs_sampling(v0)

        # Compute gradients for the RBM parameters
        positive_gradient = tf.matmul(tf.transpose(v0), h0_prob)
        negative_gradient = tf.matmul(tf.transpose(v1), h1_prob)

        # Update the weights and biases
        delta_W = positive_gradient - negative_gradient
        delta_visible_bias = tf.reduce_mean(v0 - v1, axis=0)
        delta_hidden_bias = tf.reduce_mean(h0 - h1, axis=0)

        self.W.assign_add(self.learning_rate * delta_W)
        self.visible_bias.assign_add(self.learning_rate * delta_visible_bias)
        self.hidden_bias.assign_add(self.learning_rate * delta_hidden_bias)

        return v1_prob


def train_rbm(train_data, num_hidden_units, learning_rate, num_epochs):
    num_visible_units = train_data.shape[1]

    rbm = RBM(num_visible_units, num_hidden_units)
    rbm.learning_rate = learning_rate

    for epoch in range(num_epochs):
        for batch in train_data:
            batch = tf.convert_to_tensor(batch, dtype=tf.float32)
            _ = rbm(batch)

    return rbm
def Model_RBM_Feat(train_data, train_target):
    num_hidden_units = 10
    learning_rate = 0.01
    num_epochs = 10

    # Create an RBM model
    rbm = train_rbm(train_data, num_hidden_units, learning_rate, num_epochs)

    inp = rbm.input  # input placeholder
    outputs = [layer.output for layer in rbm.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    layerNo = -2
    data = np.append(train_data, train_data, axis=0)
    Feats = []
    for i in range(data.shape[0]):
        test = data[i, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feats.append(layer_out)
    Feat = np.asarray(Feats)
    return Feat
