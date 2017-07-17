from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def _log(setting, message=''):
    output = '----------------'
    for key in setting:
        output += key + '=' + setting[key] + '--------'
    output += '\t' + message
    print(output)

def train_knn_model(X_train, X_val, y_train, y_val, n_neighbors):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    score = knn_model.score(X_val, y_val)
    _log({'n_neigbors': str(n_neighbors), 'score': str(score)})
    return {'model': knn_model, 'score': score}

def train_boosting_model(X_train, X_val, y_train, y_val, n_estimators, learning_rate):
    boosting_model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    boosting_model.fit(X_train, y_train)
    score = boosting_model.score(X_val, y_val)
    _log({'n_estimators': str(n_estimators), 'learning_rate': str(learning_rate), 'score': str(score)})
    return {'model' : boosting_model, 'score' : score}

def train_svc_model(X_train, X_val, y_train, y_val, C, gamma):
    svc_model = SVC(C=C, gamma=gamma)
    svc_model.fit(X_train, y_train)
    score = svc_model.score(X_val, y_val)
    _log({'C': str(C), 'gamma': str(gamma), 'score': str(score)})
    return {'model': svc_model, 'score': score}
