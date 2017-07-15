import numpy as np
import pandas as ps

from seqmod.preprocessing import sequentialize_vectors_non_labelled, sequentialize_vectors
from seqmod.model import SequentialSensoryDataModel
from seqmod.metrics import ScoreMap

import matplotlib.pyplot as plt

if __name__ == '__main__':

    SEQUENCE_LENGTH = 50
    DICTIONARY = ["None", "Standing", "Sitting", "Lying", "Walking", "Climbing stairs", "Waist bending", "Arm elevation", "Knees bending", "Cycling", "Jogging", "Running", "Jumping"]
    SCOREMAP = [0, 2, 1, 0, 3, 5, 4, 4, 4, 6, 4, 7, 5]

    health_scorer = ScoreMap(DICTIONARY, SCOREMAP, strategy='exponential')
    model = SequentialSensoryDataModel(path='sequential_sensory_data_model_second_iteration.bin')

    use_datasets = np.linspace(1, 10, 10)

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

        labels = []
        sizes = []
        for activity in np.unique(prediction):
            weights[activity] = len(list(filter(lambda x: x == activity, prediction))) / count
            if weights[activity]*100 > 2:
                labels.append(health_scorer.get_activitiy(activity))
                sizes.append(int(weights[activity]*100))

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        total_score = 0
        for key in weights:
            score = health_scorer.get_score(key)
            total_score += score * weights[key]
        print(total_score)

        plt.savefig('./plots/pies/subject' + str(int(dataset)) + '_score' + str(int(total_score)) + '.png')
        plt.cla()
        plt.clf()
    pass