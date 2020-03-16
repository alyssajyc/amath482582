from keras.models import Sequential
import numpy as np
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.core import  Activation
from keras.layers.core import Dense,Dropout
from keras.layers import MaxPooling2D
from keras.layers.core import Flatten
from keras import backend as K
from keras import regularizers
from keras.layers.normalization import BatchNormalization
import argparse
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import keras

from keras.utils.vis_utils import plot_model
def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()
def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)
    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(i, loss[i], acc[i], val_loss[i], val_acc[i]))
def build():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu', padding='same'))
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dropout(rate=0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1500,activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(10, activation='softmax'))
    return model

def get_data(path):
    import scipy.io as scio
    data = scio.loadmat(path)
    x_train = np.array(data["X_train"]).astype('float32').reshape(60000,28,28,1)/255.0
    y_train = np.array(data["y_train"]).T
    x_test = np.array(data["X_test"]).astype('float32').reshape(10000,28,28,1)/255.0
    y_test = np.array(data["y_test"]).T
    y_label_train_OneHot = np_utils.to_categorical(y_train)
    y_label_test_OneHot = np_utils.to_categorical(y_test)
    return (x_train,y_label_train_OneHot),(x_test,y_label_test_OneHot)


def main(path):
    (x_train,y_label_train_OneHot),(x_test,y_label_test_OneHot) = get_data(path)
    model = build()
    print(model.summary())
    try:
        model.load_weights("model/cnn", "model.h5")
        print("successful! Continue this model")
    except:
        print("Please try a new model!")

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.adam(),
                  metrics=['accuracy'])  
    plot_model(model, show_shapes=True, to_file=os.path.join("model/cnn", 'model.png'))

    print('[INFO] training network...')
    H = model.fit(x_train, y_label_train_OneHot, validation_split=0.2, batch_size=100, epochs=20, verbose=2)
    model_json = model.to_json()
    with open(os.path.join("model/cnn", 'model.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join("model/cnn", 'model.h5'))

    loss, acc = model.evaluate(x_test, y_label_test_OneHot, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    plot_history(H, "model/cnn")
    save_history(H, "model/cnn")
    prediction = model.predict_classes(x_test)  
    print(prediction.shape)
    y_label = np.array([np.argmax(label) for label in y_label_test_OneHot])
    print(y_label.shape)

    import pandas as pd
    result = pd.crosstab(y_label, prediction, rownames=['label'], colnames=['prediction'])
    print(result)


if __name__ == '__main__':
    path = "data/fashion_mnist.mat"
    main(path)

