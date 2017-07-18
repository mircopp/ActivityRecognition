import pickle as pc

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from seqmod.training import _log, train_knn_model, train_boosting_model, train_svc_model

from joblib import Parallel, delayed

class SequentialSensoryDataModel(BaseEstimator, ClassifierMixin):

    def __init__(self, path=None):
        """
        Initializes the sequential sensory data model

        :param model: If there is already an existing model please pass it here
        :param model_description: What is the model about
        """

        self.models = []
        if path:
            self.load_model(path)
        else:
            self.best_performing_model = None

    def fit(self, X, y):

        """
        Fits the sequentialized features according to an SVC, Gradient boosting or KNN model if not speficied.

        :param X: Sequential list of features for training
        :param y: Class (output) of sensory data
        :return: None
        """

        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.75, stratify=y)

        # Normalization
        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(X_train)
        X_train, X_val = self.standard_scaler.transform(X_train), self.standard_scaler.transform(X_val)

        if not self.best_performing_model:

            # Find the best model using brutforce
            # Try KNN
            knn_models = Parallel(n_jobs=-1)(delayed(train_knn_model)(X_train, X_val, y_train, y_val, n_neighbors) for n_neighbors in range(1, 81))
            best = max(knn_models, key=lambda x: x['score'])
            knn_model = best['model']
            _log({'best_score': str(best['score'])}, '\n')
            self.models.append({
                'model': knn_model,
                'description': 'knn'
            })

            # Try boosting
            boosting_models = Parallel(n_jobs=-1)(delayed(train_boosting_model)(X_train, X_val, y_train, y_val, n_estimators, learning_rate) for n_estimators in [2 ** i for i in range(0, 9)] for learning_rate in [10 ** i for i in range(-5, 2)])
            best = max(boosting_models, key=lambda x: x['score'])
            boosting_model = best['model']
            _log({'best_score' : str(best['score'])}, '\n')
            self.models.append({
                'model' : boosting_model,
                'description' : 'boosting'
            })

            # Try SVC
            svc_models = Parallel(n_jobs=-1)(delayed(train_svc_model)(X_train, X_val, y_train, y_val, C, gamma) for C in [10 ** i for i in range(-6, 6)] for gamma in [10 ** i for i in range(-6, 2)])
            best = max(svc_models, key=lambda x: x['score'])
            svc_model = best['model']
            _log({'best_score' : str(best['score'])}, '\n')
            self.models.append({
                'model': svc_model,
                'description': 'svc'
            })

            max_score = -1
            for model in self.models:
                score = model['model'].score(X_val, y_val)
                print(model['description'] + ':', score)
                self.best_performing_model = model if score > max_score else self.best_performing_model
                max_score = score if score > max_score else max_score
        else:
            self.best_performing_model['model'].fit(X_train, y_train)
            score = self.best_performing_model['model'].score(X_val, y_val)
            print(self.best_performing_model['description'] + ':', score)
        return self

    def predict(self, X):
        """
        Predicts for given features with the best performing model

        :param X: Matrix of feautures
        :return: Vector containing predicted values
        """

        return self.best_performing_model['model'].predict(X)

    def score(self, X, y, sample_weight=None):
        """
        Computes the score of the model using the best performing model

        :param X: Input matrix of features
        :param y: Real values
        :param sample_weight: Sample weight
        :return: The accurracy of the model
        """
        return self.best_performing_model['model'].score(X, y)

    def save_model(self, path='sequential_sensory_data_model.bin'):
        """
        Saves the best performing model and the others used for the model selection

        :param path: Filepath
        :return: None
        """

        pc.dump((self.best_performing_model, self.models, self.standard_scaler), open(path, 'wb'))

    def load_model(self, path='sequential_sensory_data_model.bin'):
        """
        Loads the model saved before

        :param path: Path of the model
        :return: self
        """

        self.best_performing_model, self.models, self.standard_scaler = pc.load(open(path, 'rb'))
        return self

    def normalize(self, X):
        """
        Normalizes the data accoriding to the fitting before
        :param X: The input matrix
        :return: Normalized matrix
        """

        return self.standard_scaler.transform(X)

    def get_models(self):
        """
        Get the other models beside the best model
        :return: The models
        """

        return self.models

    def set_model(self, type):
        """
        Sets another model as best model
        :param type: type of the new model in [knn, svc, boosting]
        :return: None
        """

        for model in self.models:
            if model['description'] == type:
                self.best_performing_model = model