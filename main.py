import pandas as ps
import numpy as np
import pickle as pc

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


from preprocessing.csv_transform import transform_tab_sep_to_csv
from preprocessing import compress_data

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.multiclass import OneVsRestClassifier
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
                model['model'].fit(X, y)

            max_score = -1
            for model in self.models:
                score = model['model'].score(X_val, y_val)
                print(model['description'] + ':', score)
                self.best_performing_model = model if score > max_score else self.best_performing_model
                max_score = score if score > max_score else max_score

        else:
            self.best_performing_model['model'].fit(X_train, y_train)
            score = self.best_performing_model['model'].score(X_val, y_val)
            print(score)

        return self

    def predict(self, X):
        return self.best_performing_model['model'].predict(X)

    def score(self, X, y, sample_weight=None):
        return self.best_performing_model['model'].score(X, y)

    def save_model(self, path=None):
        if path:
            pc.dump((self.best_performing_model, self.models), open(path, 'wb'))
        else:
            pc.dump((self.best_performing_model, self.models), open('sequential_sensory_data_model.bin', 'wb'))

    def load_model(self, path='sequential_sensory_data_model.bin'):
        self.best_performing_model, self.models = pc.load(open(path, 'rb'))
        return self

    def get_models(self):
        return self.models




# Main code

def train_model(sequence_length=50, train_datasets=np.linspace(1, 8, 8), test_datasets=np.linspace(9, 10, 2)):

    model = SequentialSensoryDataModel()

    # Preparation variables
    RAW_FILES_TRAINING = ['data/mHealth_subject' + str(int(i)) + '.log' for i in train_datasets]
    RAW_FILES_TEST = ['data/mHealth_subject' + str(int(i)) + '.log' for i in test_datasets]

    CSV_FILES_TRAINING = ['csv/mHealth_subject' + str(int(i)) + '.csv' for i in train_datasets]
    CSV_FILES_TEST = ['csv/mHealth_subject' + str(int(i)) + '.csv' for i in test_datasets]

    PREPARED = True

    if not PREPARED:
        for i in range(len(RAW_FILES_TRAINING)):
            transform_tab_sep_to_csv(RAW_FILES_TRAINING[i], CSV_FILES_TRAINING[i])
        for i in range(len(RAW_FILES_TEST)):
            transform_tab_sep_to_csv(RAW_FILES_TEST[i], CSV_FILES_TEST[i])

    # Load the training data
    training_data = np.array([])
    for file in CSV_FILES_TRAINING:
        data = ps.read_csv(file, sep=',').as_matrix()
        data = compress_data.sequentialize_vectors(data, sequence_length=sequence_length)
        training_data = np.vstack((training_data, data)) if training_data.any() else np.vstack(data)
    X_train, y_train = training_data[:, :-1], training_data[:, -1]

    # Load the test data
    test_data = np.array([])
    for file in CSV_FILES_TEST:
        data = ps.read_csv(file, sep=',').as_matrix()
        data = compress_data.sequentialize_vectors(data, sequence_length=sequence_length)
        test_data = np.vstack((test_data, data)) if test_data.any() else np.vstack(data)
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # Fit the model to the training dataset
    model.fit(X_train, y_train)

    # # Test the model with best parameter and test data
    print('\nTest score:', model.score(X_test, y_test))

    return model

if __name__ == '__main__':
    print('Run model training:\n')
    size = 50
    model = train_model(sequence_length=size)

    print('Save best model...\n')
    model.save_model()

    print('Done!')

