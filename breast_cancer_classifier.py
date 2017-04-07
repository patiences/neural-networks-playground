import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten

class BreastCancerData():
    def __init__(self):
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        print(self.X_train.shape[0], 'training samples')
        print(self.X_test.shape[0], 'test samples')

        # Perform scaling
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

def sk_learn_nn(data):
    # 3 hidden layers with 30 units each
    model = MLPClassifier(hidden_layer_sizes=(30,30,30)) # relu activation
    model.fit(data.X_train, data.y_train)

    print('scikit-learn neural net:')
    eval(model, data)

# FIXME dimensions of internal layers?
def tensorflow_nn(data):
    model = Sequential()

    model.add(Dense(30, input_shape=(data.X_train.shape[1],), activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    # Output layer is 2 classes
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrix=['accuracy'])
    history = model.fit(data.X_train, data.y_train, batch_size=128, epochs=10, verbose=0)

    print('tensorflow neural net:')

    score = model.evaluate(data.X_train, data.y_train, verbose=0)
    print('Training accuracy:', score[1])

    score = model.evaluate(data.X_test, data.y_test, verbose=0)
    print('Test accuracy:', score[1])

def random_forest_classifier(data):
    model = RandomForestClassifier()
    model.fit(data.X_train, data.y_train)

    print('Random forest classifier:')
    eval(model, data)

def eval(model, data):
    score = model.score(data.X_train, data.y_train)
    print('     Training accuracy:', score)

    score = model.score(data.X_test, data.y_test)
    print('     Test accuracy:', score)

    y_hat = model.predict(data.X_test)

    print("Confusion matrix: ")
    print(confusion_matrix(data.y_test, y_hat))
    print("Classification report: ")
    print(classification_report(data.y_test, y_hat))

if __name__ == '__main__':
    data = BreastCancerData()

    sk_learn_nn(data)
    tensorflow_nn(data)
    random_forest_classifier(data)
