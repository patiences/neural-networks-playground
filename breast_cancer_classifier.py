import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(X_train.shape[0], 'training samples')
    print(X_test.shape[0], 'test samples')

    # Perform scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # 3 hidden layers with 30 units each
    model = MLPClassifier(hidden_layer_sizes=(30,30,30)) # relu activation
    model.fit(X_train,y_train)

    # Compute training error
    yhat = model.predict(X_train)
    trainError = np.sum(y_train!=yhat) / float(yhat.size)
    print("Training error = ", trainError)

    # Compute test error
    yhat = model.predict(X_test)
    testError = np.sum(y_test!=yhat) / float(yhat.size)
    print("Test error     = ", testError)

    # Model evaluation
    print("Confusion matrix: ")
    print(confusion_matrix(y_test,yhat))
    print("Classification report: ")
    print(classification_report(y_test,yhat))

if __name__ == '__main__':
    main()
