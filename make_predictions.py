import pickle as pc
import numpy as np
import pandas as ps
import matplotlib.pyplot as plt
from preprocessing import compress_data

from main import SequentialSensoryDataModel


if __name__ == '__main__':


    sequence_length = 50
    model = SequentialSensoryDataModel(path='sequential_sensory_data_model.bin')

    use_datasets = np.linspace(1, 10, 10)

    for dataset in use_datasets:

        CSV_FILES = ['csv/mHealth_subject' + str(int(dataset)) + '.csv']

        matrix = np.array([])
        # Load data
        for file in CSV_FILES:
            data = ps.read_csv(file, sep=',').as_matrix()
            data = compress_data.sequentialize_vectors_non_labelled(data, sequence_length=sequence_length)
            matrix = np.vstack((matrix, data)) if matrix.any() else np.vstack(data)

        matrix = matrix[:, :-1]

        prediction = model.predict(matrix)
        # plt.hist(prediction, bins=np.linspace(0.5,12.5,13))
        # plt.show()

        count = len(prediction)
        weights = {}
        for activity in np.unique(prediction):
            weights[activity] = len(list(filter(lambda x: x == activity, prediction))) / count

        total_score = 0
        for key in weights:
            score = compress_data.get_score(key)
            total_score += score * weights[key]
        print(total_score)

    pass