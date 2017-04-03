from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.neural_network import MLPClassifier

""" Code adapted from https://ubc-cs.github.io/cpsc340/ """

class MnistData:
    def __init__(self):
        self.img_dim = (28,28)
        self.img_size = self.img_dim[0] * self.img_dim[1]
        # 10 digits
        self.num_classes = 10

        # the data, shuffled and split between train and test sets
        (self.X_train, self.y_train_cat), (self.X_test, self.y_test_cat) = mnist.load_data()

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255
        self.X_train_flat = self.X_train.reshape(60000, self.img_size)
        self.X_test_flat = self.X_test.reshape(10000, self.img_size)

        self.X_train = self.X_train[...,None] # add 4th dimension for convnets
        self.X_test = self.X_test[...,None]

        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(self.y_train_cat, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test_cat, self.num_classes)

        print(self.X_train.shape[0], 'training samples')
        print(self.X_test.shape[0], 'test samples')

def sk_learn_nn(data):
    X_train_flat = data.X_train_flat
    y_train_cat = data.y_train_cat

    X_test_flat = data.X_test_flat
    y_test_cat = data.y_test_cat

    # 2 hidden layers with 100 and 50 units respectively
    model = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=15, batch_size=128)
    model.fit(X_train_flat, y_train_cat)

    print("With Scikit-Learn NN: ")

    score = model.score(X_train_flat, y_train_cat)
    print('Training accuracy:', score)

    score = model.score(X_test_flat, y_test_cat)
    print('Test accuracy:', score)

def tensorflow_nn(data):
    X_train_flat = data.X_train_flat
    y_train = data.y_train

    X_test_flat = data.X_test_flat
    y_test = data.y_test

    model = Sequential() # as opposed to recurrent neural nets
    # Need to specify input layer dimensions for compilation step
    # This is 2 layers with 100 units each
    model.add(Dense(100, input_shape=(X_train_flat.shape[1],), activation='relu'))
    model.add(Dense(100, activation='relu'))
    # Output layer is 10 classes
    model.add(Dense(data.num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train_flat, y_train, batch_size=128, epochs=10, verbose=0)

    print("With tensorFlow NN: ")

    score = model.evaluate(X_train_flat, y_train, verbose=0)
    print('Training accuracy:', score[1])

    score = model.evaluate(X_test_flat, y_test, verbose=0)
    print('Test accuracy:', score[1])

def tensorflow_convnet(data):
    X_train = data.X_train
    y_train = data.y_train
    X_test = data.X_test
    y_test = data.y_test

    model = Sequential()
    # 32 5x5 convolutions
    model.add(Convolution2D(32, 5, 5, input_shape=data.img_dim+(1,), activation='relu'))
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
    data = MnistData()

    sk_learn_nn(data)
    tensorflow_nn(data)
    tensorflow_convnet(data)
