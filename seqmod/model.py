import pickle as pc

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

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

        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, stratify=y)

        if not self.best_performing_model:
            # Find the best model using brutforce

            # Try KNN
            pipe_knn = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier()
            )
            knn_model = GridSearchCV(
                estimator=pipe_knn,
                param_grid=[{
                    "kneighborsclassifier__n_neighbors": range(1, 81)
                }],
                verbose=2,
                n_jobs=-1
            )
            self.models.append({
                'model': knn_model,
                'description': 'knn'
            })

            # Try boosting
            pipe_boosting = make_pipeline(
                StandardScaler(),
                GradientBoostingClassifier()
            )
            boosting_model = GridSearchCV(
                estimator=pipe_boosting,
                param_grid=[{
                    "gradientboostingclassifier__n_estimators": [2 ** i for i in range(0, 8)],
                    "gradientboostingclassifier__learning_rate": [10 ** i for i in range(-5, 1)]
                }],
                verbose=2,
                n_jobs=-1
            )
            self.models.append({
                'model' : boosting_model,
                'description' : 'boosting'
            })

            # Try SVC
            pipe_svc = make_pipeline(
                StandardScaler(),
                SVC()
            )
            svc_model = GridSearchCV(
                estimator=pipe_svc,
                param_grid=[{
                    "svc__C": [10 ** i for i in range(-2, 5)],
                    "svc__gamma": [10 ** i for i in range(-4, 2)]
                }],
                verbose=2,
                n_jobs=-1
            )
            self.models.append({
                'model': svc_model,
                'description': 'svc'
            })


            for model in self.models:
                model['model'].fit(X_train, y_train)

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
        return self.best_performing_model['model'].predict(X)

    def transform(self, X):
        return self.best_performing_model['model'].transform(X)

    def score(self, X, y, sample_weight=None):
        return self.best_performing_model['model'].score(X, y)

    def save_model(self, path='sequential_sensory_data_model.bin'):
        pc.dump((self.best_performing_model, self.models), open(path, 'wb'))

    def load_model(self, path='sequential_sensory_data_model.bin'):
        self.best_performing_model, self.models = pc.load(open(path, 'rb'))
        return self

    def get_models(self):
        return self.models