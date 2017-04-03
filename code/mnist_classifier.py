from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.neural_network import MLPClassifier

""" Code taken from https://ubc-cs.github.io/cpsc340/ """

def main():
    # the data, shuffled and split between train and test sets
    (X_train, y_train_cat), (X_test, y_test_cat) = mnist.load_data()

    img_dim = (28,28)
    img_size = img_dim[0]*img_dim[1]
    # 10 digits
    num_classes = 10

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_train_flat = X_train.reshape(60000, img_size)
    X_test_flat = X_test.reshape(10000, img_size)

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train_cat, num_classes)
    y_test = np_utils.to_categorical(y_test_cat, num_classes)

    print(X_train.shape[0], 'training samples')
    print(X_test.shape[0], 'test samples')

    # 2 hidden layers with 100 and 50 units respectively
    nn = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=15, batch_size=128)
    nn.fit(X_train_flat, y_train_cat)

    score = nn.score(X_train_flat, y_train_cat)
    print('Training accuracy:', score)

    score = nn.score(X_test_flat, y_test_cat)
    print('Test accuracy:', score)

if __name__ == '__main__':
    main()
