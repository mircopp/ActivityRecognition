import pandas as ps
import numpy as np
from sklearn.model_selection import train_test_split

from actreg.seqmod import SequentialSensoryDataModel

def train_model(sequence_length=50, train_datasets=np.linspace(1, 9, 9), test_datasets=np.linspace(10, 10, 1), save_path='sequential_sensory_data_model.bin'):

    print('Run model training:\n')
    model = SequentialSensoryDataModel(sequence_length=sequence_length)

    TRAINING_FILES = ['data_collection/labelled/mHealth_subject' + str(int(i)) + '.csv' for i in train_datasets]
    TEST_FILES = ['data_collection/labelled/mHealth_subject' + str(int(i)) + '.csv' for i in test_datasets]

    # Load the training data
    X_train, y_train = np.array([]), np.array([])
    for file in TRAINING_FILES:
        XY_train = ps.read_csv(file, sep=',').as_matrix()
        X_tmp_train, y_tmp_train = model.sequentialize(XY_train[:, :-1], XY_train[:, -1])
        X_train = np.append(X_train, X_tmp_train, axis=0) if X_train.any() else X_tmp_train
        y_train = np.append(y_train, y_tmp_train) if y_train.any() else y_tmp_train

    # Load the test data
    X_test, y_test = np.array([]), np.array([])
    for file in TEST_FILES:
        XY_test = ps.read_csv(file, sep=',').as_matrix()
        X_tmp_test, y_tmp_test = model.sequentialize(XY_test[:, :-1], XY_test[:, -1])
        X_test = np.append(X_test, X_tmp_test, axis=0) if X_test.any() else X_tmp_test
        y_test = np.append(y_test, y_tmp_test) if y_test.any() else y_tmp_test

    # Fit the model to the training dataset
    model.fit(X_train, y_train)

    # Test the model with best parameter and normalized test data
    X_test = model.normalize(X_test)
    print('\nTest score:', model.score(X_test, y_test))

    # Safe the model and all his components
    print('Save best model...\n')
    model.save_model(path=save_path)

    print('Done!')

def train_model_classical(sequence_length=50, data_sets=np.linspace(1, 10, 10), save_path='sequential_sensory_data_model_classical.bin'):

    print('Run model training (classical strategy):\n')
    model = SequentialSensoryDataModel(sequence_length=sequence_length)

    FILES = ['data_collection/labelled/mHealth_subject' + str(int(i)) + '.csv' for i in data_sets]

    # Load the data
    X, y = np.array([]), np.array([])
    for file in FILES:
        XY = ps.read_csv(file, sep=',').as_matrix()
        X_tmp, y_tmp = model.sequentialize(XY[:, :-1], XY[:, -1])
        X = np.append(X, X_tmp, axis=0) if X.any() else X_tmp
        y = np.append(y, y_tmp) if y.any() else y_tmp

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, stratify=y)

    # Fit the model to the training dataset
    model.fit(X_train, y_train)

    # Test the model with best parameter and normalized test data
    X_test = model.normalize(X_test)
    print('\nTest score:', model.score(X_test, y_test))

    # Safe the model and all his components
    print('Save the model...\n')
    model.save_model(path=save_path)

    print('Done!')

if __name__ == '__main__':
    size = 50
    train_model_classical(sequence_length=size)
    train_model(sequence_length=size)