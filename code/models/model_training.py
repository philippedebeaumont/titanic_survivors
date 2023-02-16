from sklearn.svm import SVC


def train_model(X_train, y_train, model):
    svc = SVC()
    svc.fit(X_train, y_train)

    return svc