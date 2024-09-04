import keras
from keras import layers
from keras import backend as K
import numpy as np

def Model_Autoencoder_Feat(Train_Data, Train_Target):
    # This is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = keras.Input(shape=(Train_Data.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(Train_Target.shape[1], activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    Train_Data = Train_Data.astype('float32') / 255.
    Train_Data = Train_Data.reshape((len(Train_Data), np.prod(Train_Data.shape[1:])))
    autoencoder.fit(Train_Data, Train_Target,
                    epochs=50,
                    batch_size=256,
                    shuffle=True)

    inp = autoencoder.input  # input placeholder
    outputs = [layer.output for layer in autoencoder.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    layerNo = -1
    data = np.append(Train_Data, Train_Data, axis=0)
    Feats = []
    for i in range(data.shape[0]):
        test = data[i, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feats.append(layer_out)
    Feature = np.asarray(Feats)
    return Feature
