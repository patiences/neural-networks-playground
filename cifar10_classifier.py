import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import sys
import h5py

class Cifar10Data:
    def __init__(self, debug=False):
        self.img_dim = (32,32)
        self.img_size = self.img_dim[0] * self.img_dim[1]
        # 10 categories
        self.num_classes = 10

        # the data, shuffled and split between train and test sets
        (self.X_train, self.y_train_cat), (self.X_test, self.y_test_cat) = cifar10.load_data()

        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(self.y_train_cat, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test_cat, self.num_classes)

        # data augmentation
        self.datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        self.datagen.fit(self.X_train)

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255

        # n by H*W*3 matrix
        self.X_train_flat = self.X_train.reshape(50000, self.img_size*3)
        self.X_test_flat = self.X_test.reshape(10000, self.img_size*3)

        if debug == True:
            # for debugging with smaller samples
            num_training = 5000
            num_test = 10000

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

    print("Scikit-Learn NN: ")
    print("     Model: ", model)

    model.fit(X_train_flat, y_train_cat)

    score = model.score(X_train_flat, y_train_cat)
    print('     Training accuracy:', score)

    score = model.score(X_test_flat, y_test_cat)
    print('     Test accuracy:', score)

def tensorflow_nn(data):
    X_train_flat = data.X_train_flat
    y_train = data.y_train

    X_test_flat = data.X_test_flat
    y_test = data.y_test

    model = Sequential()
    # 2 layers with 100 units each
    model.add(Dense(100, input_shape=(X_train_flat.shape[1],), activation='relu'))
    model.add(Dense(100, activation='relu'))
    # Output layer is 10 classes
    model.add(Dense(data.num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train_flat, y_train, batch_size=128, epochs=10, verbose=1)

    print("TensorFlow NN: ")
    print("     Model: ", model.summary())

    score = model.evaluate(X_train_flat, y_train, verbose=1)
    print('     Training accuracy:', score[1])

    score = model.evaluate(X_test_flat, y_test, verbose=1)
    print('     Test accuracy:', score[1])

def tensorflow_convnet(data):
    X_train = data.X_train
    y_train = data.y_train
    X_test = data.X_test
    y_test = data.y_test

    model = Sequential()
    # 32 5x5 convolutions
    model.add(Convolution2D(32, (5, 5), input_shape=data.img_dim+(3,), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))
    model.add(Flatten())
    # Hidden layer with 128 units
    model.add(Dense(128, activation='relu'))
    # Output layer is the num classes
    model.add(Dense(data.num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # checkpoint -- save the state of the model as we go
    filepath="cifar10-checkpoints/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    callbacks = [checkpoint]

    # Trains the model, using generated batches of augmented data for training and validation.
    history = model.fit_generator(data.datagen.flow(X_train, y_train, batch_size=32),
                # validation_steps = # unique samples in training set / batch_size https://keras.io/models/sequential/#fit_generator
                validation_data=data.datagen.flow(X_train, y_train, batch_size=32), validation_steps=len(data.X_train)/32,
                steps_per_epoch=len(X_train), epochs=60, callbacks=callbacks, verbose=1)

    # plot validation accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['train', 'test'])
    plt.savefig('accuracy.pdf')

    print("TensorFlow ConvNet:")
    print("     Model: ", model.summary())

    score = model.evaluate(X_train, y_train, verbose=1)
    print('     Training accuracy:', score[1])

    score = model.evaluate(X_test, y_test, verbose=1)
    print('     Test accuracy:', score[1])

def random_forest_classifier(data):
    model = RandomForestClassifier()
    model.fit(data.X_train_flat, data.y_train_cat)

    print('Random forest classifier:')
    print("     Model: ", model)

    score = model.score(data.X_train_flat, data.y_train_cat)
    print('     Training accuracy:', score)

    score = model.score(data.X_test_flat, data.y_test_cat)
    print('     Test accuracy:', score)

if __name__ == '__main__':

    # Allow debugging with small sample size
    debug = True if len(sys.argv) > 1 and sys.argv[1] == 'debug' else False

    data = Cifar10Data(debug=debug)

    random_forest_classifier(data)
    sk_learn_nn(data)
    tensorflow_nn(data)
    tensorflow_convnet(data)
    print("==============================================================================")

    # Cleanup for keras: https://github.com/tensorflow/tensorflow/issues/3388
    import gc
    gc.collect()
    from keras import backend
    backend.clear_session()
