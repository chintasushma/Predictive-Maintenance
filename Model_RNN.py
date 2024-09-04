import numpy as np
from Evaluation import evaluation
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
def Model_RNN(train_data, train_target, test_data, test_target):
    out, model = RNN_train(train_data, train_target, test_data)

    Eval = evaluation(out.astype('int'), test_target)
    return np.asarray(Eval).ravel()

def RNN_train(trainX, trainY, testX):
    trainX = np.resize(trainX, (trainX.shape[0], 1, 200))
    testX = np.resize(testX, (testX.shape[0], 1, 200))
    model = Sequential()
    model.add(LSTM(int(25), input_shape=(1, 200)))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)
    testPredict = model.predict(testX)
    return testPredict, model

