import pandas as ps
import numpy as np
from sklearn.model_selection import train_test_split

from actreg.seqmod import SequentialSensoryDataModel

def execute_model_training (sequence_length=50, data_sets=np.linspace(1, 10, 10), save_path=None):
    """
    Runs the  model training of SequentialSensoryDataModel with given vital sign data
    :param sequence_length: The length of the sequentialization
    :param data_sets: The bumbers of datasets to use
    :param save_path: The storage path were the final model should be stored
    :return: None
    """

    print('Execute model training:\n')
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
    if save_path:
        model.save_model(path=save_path)
    else:
        model.save_model()

    print('Done!')

if __name__ == '__main__':
    execute_model_training()