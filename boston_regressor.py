import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense
from keras.wrappers.scikit_learn import KerasRegressor

class BostonData():
    def __init__(self):
        X, y = load_boston(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        print(self.X_train.shape[0], 'training samples')
        print(self.X_test.shape[0], 'test samples')

        # Perform scaling on X
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)


def sk_learn_nn(data):
    model = MLPRegressor(hidden_layer_sizes=(150,125,20), activation='relu')
    model.fit(data.X_train, data.y_train)

    print('scikit-learn neural net:')
    eval(model, data)

def tensorflow_nn(data):
    model = Sequential()

    model.add(Dense(100, input_shape=(data.X_train.shape[1],), activation='relu'))
    model.add(Dense(125, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_absolute_error', optimizer='adam')
    history = model.fit(data.X_train, data.y_train, batch_size=64, epochs=20, verbose=0)

    print('tensorflow neural net:')

    score = model.evaluate(data.X_train, data.y_train, verbose=0)
    print('     Training accuracy:', score)

    score = model.evaluate(data.X_test, data.y_test, verbose=0)
    print('     Test accuracy:', score)

def linear_regressor(data):
    model = LinearRegression()
    model.fit(data.X_train, data.y_train)

    print('linear regression:')
    eval(model, data)

def ridge_regressor(data):
    model = Ridge(alpha=0.5)
    model.fit(data.X_train, data.y_train)

    print('ridge regression:')
    eval(model, data)

def bayesian_ridge_regressor(data):
    model = BayesianRidge()
    model.fit(data.X_train, data.y_train)

    print('Bayesian ridge regression:')
    eval(model, data)

def random_forest_regressor(data):
    model = RandomForestRegressor()
    model.fit(data.X_train, data.y_train)

    print('Random Forest Regressor:')
    eval(model, data)

def eval(model, data):
    score = model.score(data.X_train, data.y_train)
    print('     Training accuracy:', score)

    score = model.score(data.X_test, data.y_test)
    print('     Test accuracy:', score)

if __name__ == '__main__':
    data = BostonData()

    linear_regressor(data)
    ridge_regressor(data)
    bayesian_ridge_regressor(data)
    random_forest_regressor(data)
    sk_learn_nn(data)
    tensorflow_nn(data)
