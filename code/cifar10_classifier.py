from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.neural_network import MLPClassifier
import numpy as np

class Cifar10Data:
    def __init__(self, debug=False):
        self.img_dim = (32,32)
        self.img_size = self.img_dim[0] * self.img_dim[1]
        # 10 categories
        self.num_classes = 10

        # the data, shuffled and split between train and test sets
        (self.X_train, self.y_train_cat), (self.X_test, self.y_test_cat) = cifar10.load_data()

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255

        # H*W*n by 3 matrix
        self.X_train_flat = self.X_train.reshape(50000, self.img_size*3)
        self.X_test_flat = self.X_test.reshape(10000, self.img_size*3)

        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(self.y_train_cat, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test_cat, self.num_classes)

        if debug == True:
            # for debugging with smaller samples,
            num_training = 100
            num_test = 10

            # Choose different examples (without replacement)
            training_sample_indices = np.random.choice(50000, num_training, replace=False)
            test_sample_indices = np.random.choice(10000, num_test, replace=False)

            self.X_train = self.X_train[training_sample_indices]
            self.X_test = self.X_test[test_sample_indices]
            self.X_train_flat = self.X_train_flat[training_sample_indices]
            self.X_test_flat = self.X_test_flat[test_sample_indices]
            self.y_train_cat = self.y_train_cat[training_sample_indices]
            self.y_test_cat = self.y_test_cat[test_sample_indices]
            self.y_train = self.y_train[training_sample_indices]
            self.y_test = self.y_test[test_sample_indices]

        print(self.X_train.shape[0], 'training samples')
        print(self.X_test.shape[0], 'test samples')

def sk_learn_nn(data):
    X_train_flat = data.X_train_flat
    y_train_cat = data.y_train_cat

    X_test_flat = data.X_test_flat
    y_test_cat = data.y_test_cat

    # 2 hidden layers with 100 and 50 units respectively
    model = MLPClassifier(hidden_layer_sizes=(175,130), max_iter=20, batch_size=128, activation='relu')
    model.fit(X_train_flat, y_train_cat)

    print("With Scikit-Learn NN: ")

    score = model.score(X_train_flat, y_train_cat)
    print('Training accuracy:', score)

    score = model.score(X_test_flat, y_test_cat)
    print('Test accuracy:', score)

def tensorflow_convnet(data):
    X_train = data.X_train
    y_train = data.y_train
    X_test = data.X_test
    y_test = data.y_test

    model = Sequential()
    # 32 5x5 convolutions
    model.add(Convolution2D(32, 5, 5, input_shape=data.img_dim+(3,), activation='relu'))
    # 2x2 max pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    # Hidden layer with 128 units
    model.add(Dense(128, activation='relu'))
    # Output layer is the num classes
    model.add(Dense(data.num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Do not increase epochs unless running on a more powerful machine
    history = model.fit(X_train, y_train, batch_size=128, epochs=1, verbose=0)

    print("With ConvNet:")

    score = model.evaluate(X_train, y_train, verbose=0)
    print('Training accuracy:', score[1])

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    data = Cifar10Data()

    sk_learn_nn(data)
    tensorflow_convnet(data)
