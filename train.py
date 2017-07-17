import pandas as ps
import numpy as np

from seqmod.preprocessing import transform_to_csv, Sequentializer
from seqmod.model import SequentialSensoryDataModel

def train_model(sequence_length=50, train_datasets=np.linspace(1, 9, 9), test_datasets=np.linspace(10, 10, 1), save_path='sequential_sensory_data_model.bin'):

    print('Run model training:\n')
    model = SequentialSensoryDataModel()

    PREPARED = True

    # Files
    RAW_FILES_TRAINING = ['data/mHealth_subject' + str(int(i)) + '.log' for i in train_datasets]
    RAW_FILES_TEST = ['data/mHealth_subject' + str(int(i)) + '.log' for i in test_datasets]

    CSV_FILES_TRAINING = ['csv/mHealth_subject' + str(int(i)) + '.csv' for i in train_datasets]
    CSV_FILES_TEST = ['csv/mHealth_subject' + str(int(i)) + '.csv' for i in test_datasets]

    if not PREPARED:
        for i in range(len(RAW_FILES_TRAINING)):
            transform_to_csv(RAW_FILES_TRAINING[i], CSV_FILES_TRAINING[i], sep='\t')
        for i in range(len(RAW_FILES_TEST)):
            transform_to_csv(RAW_FILES_TEST[i], CSV_FILES_TEST[i], sep='\t')

    # Load the training data
    training_data = np.array(list(filter(lambda x: x[-1] != 0, Sequentializer(CSV_FILES_TRAINING).transform())))

    # Load the test data
    test_data = np.array(list(filter(lambda x: x[-1] != 0, Sequentializer(CSV_FILES_TEST).transform())))

    X_train, y_train = training_data[:, :-1], training_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # Fit the model to the training dataset
    model.fit(X_train, y_train)


    # # Test the model with best parameter and normalized test data
    X_test = model.normalize(X_test)
    print('\nTest score:', model.score(X_test, y_test))

    print('Save best model...\n')
    model.save_model(path=save_path)

    print('Done!')

    return model




