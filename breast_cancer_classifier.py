import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import GridSearchCV
from keras.optimizers import SGD

class BreastCancerData():
    def __init__(self):
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        print(self.X_train.shape[0], 'training samples')
        print(self.X_test.shape[0], 'test samples')

        print(self.X_train[0])

        # Perform scaling
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

def sk_learn_nn(data):
    model = MLPClassifier()

    # hyperparameter search
    activation = ['logistic', 'tanh', 'relu']
    solver = ['adam', 'sgd', 'lbfgs']
    hidden_layer_sizes = [(30, 30, 30), (50, 50, 50), (50, 40, 30), (100, 100)]
    param_grid = dict(activation=activation, solver=solver,)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(data.X_train, data.y_train)

    print('scikit-learn neural net:')
    eval(grid_result, data)

def tensorflow_nn(data):
    model = Sequential()

    model.add(Dense(30, input_shape=(data.X_train.shape[1],), activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    # Output layer is 1
    model.add(Dense(1, activation='sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrix=['accuracy'])

    history = model.fit(data.X_train, data.y_train, batch_size=128, epochs=100, verbose=0)

    print('tensorflow neural net:')

    score = model.evaluate(data.X_train, data.y_train, verbose=0)
    print('Training accuracy:', score)

    score = model.evaluate(data.X_test, data.y_test, verbose=0)
    print('Test accuracy:', score)

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
