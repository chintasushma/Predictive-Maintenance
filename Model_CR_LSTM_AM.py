import tensorflow as tf
from keras.layers import Layer, Input, LSTM, Dense
from keras.models import Model
from Evaluation import evaluation
def Model_CR_LSTM_AM(X_train, y_train, X_test, y_test):

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # Define the attention mechanism layer
    class AttentionLayer(Layer):
        def __init__(self):
            super(AttentionLayer, self).__init__()

        def build(self, input_shape):
            self.W_q = self.add_weight(
                shape=(input_shape[-1], input_shape[-1]),
                initializer='random_normal',
                trainable=True,
            )
            self.W_k = self.add_weight(
                shape=(input_shape[-1], input_shape[-1]),
                initializer='random_normal',
                trainable=True,
            )
            super(AttentionLayer, self).build(input_shape)

        def call(self, inputs):
            q = tf.matmul(inputs, self.W_q)
            k = tf.matmul(inputs, self.W_k)
            attn_score = tf.matmul(q, k, transpose_b=True)
            attn_score = tf.nn.softmax(attn_score, axis=-1)
            return tf.matmul(attn_score, inputs)

    # Define the cascaded residual LSTM model with attention
    input_layer = Input(shape=(input_dim,))
    lstm_layer1 = LSTM(64, return_sequences=True)(input_layer)
    attention1 = AttentionLayer()(lstm_layer1)
    residual1 = tf.keras.layers.Add()([lstm_layer1, attention1])

    lstm_layer2 = LSTM(64, return_sequences=True)(residual1)
    attention2 = AttentionLayer()(lstm_layer2)
    residual2 = tf.keras.layers.Add()([lstm_layer2, attention2])

    output_layer = Dense(output_dim)(residual2)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    pred = model.predict(X_test)
    predict = pred.astype('int')
    Eval = evaluation(predict, y_test)
    return Eval
