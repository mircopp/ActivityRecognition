import numpy as np
import pandas as ps

from seqmod.preprocessing import sequentialize_vectors_non_labelled, sequentialize_vectors, DataProcessing
from seqmod.model import SequentialSensoryDataModel

if __name__ == '__main__':

    SEQUENCE_LENGTH = 50
    DICTIONARY = ["None", "Standing Still", "Sitting and relaxing", "Lying Down", "Walking", "Climbing stairs", "Waist bends forward", "Frontal elevation of arms", "Knees bending", "Cycling", "Jogging", "Running", "Jump front & back"]
    SCOREMAP = [0, 2, 1, 0, 3, 4, 4, 4, 4, 5, 6, 7, 4]

    health_scorer = DataProcessing(DICTIONARY, SCOREMAP, strategy='kubic')
    model = SequentialSensoryDataModel(path='sequential_sensory_data_model_first_iteration.bin')

    use_datasets = np.linspace(9, 10, 2)

    for dataset in use_datasets:

        CSV_FILES = ['csv/mHealth_subject' + str(int(dataset)) + '.csv']

        # Load data
        matrix = np.array([])
        for file in CSV_FILES:
            data = ps.read_csv(file, sep=',').as_matrix()
            data = sequentialize_vectors_non_labelled(data, sequence_length=SEQUENCE_LENGTH)
            matrix = np.vstack((matrix, data)) if matrix.any() else np.vstack(data)

        X, y = matrix[:, :-1], matrix[:, -1]

        prediction = model.predict(X)

        count = len(prediction)
        weights = {}
        for activity in np.unique(prediction):
            weights[activity] = len(list(filter(lambda x: x == activity, prediction))) / count

        total_score = 0
        for key in weights:
            score = health_scorer.get_score(key)
            total_score += score * weights[key]
        print(total_score)

    pass