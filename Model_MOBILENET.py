import numpy as np
from keras.applications import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import cv2 as cv
from Evaluation import evaluation
def Model_MOBILENET(train_data, y_train, test_data, y_test):
    input_shape = (224, 224, 3)
    num_classes = y_train.shape[1]

    X_train = np.zeros((train_data.shape[0], 224, 224, 3))
    for i in range(len(X_train)):
        X_train[i, :] = cv.resize(train_data[i], [224, 224, 3])

    X_test = np.zeros((test_data.shape[0], 224, 224, 3))
    for i in range(len(X_test)):
        X_test[i, :] = cv.resize(test_data[i], [224, 224, 3])

    # Create MobileNet model
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    batch_size = 32
    epochs = 10

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    pred = model.predict(X_test)
    Eval = evaluation(pred.astype('int'), y_test)
    return Eval
