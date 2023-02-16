from sklearn.metrics import accuracy_score


def score(X_test, y_test, model):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)