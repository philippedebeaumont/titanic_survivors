from sklearn.model_selection import train_test_split


def preprocess_data(df):

    df = df.dropna()

    X = df.drop(['Survived'], axis=1)
    y = df['Survived']

    X = X.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    X['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    X['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values)

    return {'X_train': X_train, 'X_test': X_test, 
            'y_train': y_train, 'y_test': y_test}