import pandas as ps
import numpy as np
import pickle as pc

from preprocessing import csv_transform
from preprocessing import compress_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

def train_model(sequence_length, use_datasets=np.linspace(1, 10, 10)):
    # Preparation variables
    RAW_FILES = ['data/mHealth_subject' + str(int(i)) + '.log' for i in use_datasets]
    CSV_FILES = ['csv/mHealth_subject' + str(int(i)) + '.csv' for i in use_datasets]

    PREPARED = True

    if not PREPARED:
        for i in range(len(RAW_FILES)):
            csv_transform.transform_tab_sep_to_csv(RAW_FILES[i], CSV_FILES[i])

    matrix = np.array([])
    # Load data
    for file in CSV_FILES:
        data = ps.read_csv(file, sep=',').as_matrix()
        data = compress_data.sequentialize_vectors(data, sequence_length=sequence_length)
        matrix = np.vstack((matrix, data)) if matrix.any() else np.vstack(data)

    X, y = matrix[:, :-1], matrix[:, -1]

    X_train, X_test, y_train, y_test =  train_test_split(X, y, train_size=0.5, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5, stratify=y_test)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    scores = []

    # # Train the model with different complexity parameters
    for c in 10 ** np.linspace(-6, 6, 13):
        for gamma in 10 ** np.linspace(-6, 2, 9):
            print('\tAnalysing C:\t', round(c, 6), '\t...\tGamma:\t', round(gamma, 6))
            model = OneVsRestClassifier(SVC(C=c, gamma=gamma), n_jobs=-1)
            model.fit(X_train, y_train)
            accuracy = model.score(X_val, y_val)  # validate the models accuracy
            json = {'c': c, 'gamma': gamma, 'accuracy': accuracy}
            print('\tResult of analysis:', round(accuracy, 6),
                  '\n----------------------------------------------------------------------------\n')
            scores.append(json)

    # # Test the model with best parameter and test data
    max_accuracy = max(scores, key=lambda x: x['accuracy'])
    print('\nBest configuration:\n', '\tC:\t', max_accuracy['c'], '\t...\tGamma:\t', max_accuracy['gamma'], '\taccuracy:', max_accuracy['accuracy'])
    max_c = max_accuracy['c']
    max_gamma = max_accuracy['gamma']
    model = SVC(C=max_c, gamma=max_gamma)
    model.fit(X_train, y_train)
    print('\nTest score:', model.score(X_test, y_test))

    return (model, sc)

if __name__ == '__main__':
    print('Run model training:\n')
    size = 100
    model, scaler = train_model(sequence_length=size, use_datasets=np.linspace(1, 10, 10))

    print('Save best model...\n')
    pc.dump((model, scaler), open('model' + str(size)  + '.bin', 'wb'))
    print('Done!')

