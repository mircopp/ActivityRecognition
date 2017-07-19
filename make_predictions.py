import numpy as np
import pandas as ps

from actreg.seqmod import SequentialSensoryDataModel
from actreg.metrics import ScoreMap

import matplotlib.pyplot as plt

if __name__ == '__main__':

    DICTIONARY = ["None", "Standing", "Sitting", "Lying", "Walking", "Climbing stairs", "Waist bending", "Arm elevation", "Knees bending", "Cycling", "Jogging", "Running", "Jumping"]
    SCOREMAP = [0, 2, 1, 0, 3, 5, 4, 4, 4, 6, 4, 7, 5]

    # Setup the score map
    score_map = ScoreMap(DICTIONARY, SCOREMAP, strategy='exponential')

    # Setup the model
    model = SequentialSensoryDataModel()
    model.load_model()

    use_datasets = np.linspace(1, 10, 10)
    for dataset in use_datasets:
        file = 'data_collection/non_labelled/mHealth_non_labelled_subject' + str(int(dataset)) + '.csv'

        # Load data
        X = ps.read_csv(file, sep=',').as_matrix()
        X = model.sequentialize(X)

        # Normalize the data
        X = model.normalize(X)

        # Predict on the data
        prediction = model.predict(X)

        # Compute weights (percentage of whole data) for the scores and plot results
        count = len(prediction)
        weights = {}
        labels = []
        sizes = []
        for activity in np.unique(prediction):
            weights[activity] = len(list(filter(lambda x: x == activity, prediction))) / count
            if weights[activity]*100 > 2:
                labels.append(score_map.get_activitiy(activity))
                sizes.append(int(weights[activity]*100))
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        total_score = 0
        for key in weights:
            score = score_map.get_score(key)
            total_score += score * weights[key]
        print(total_score)

        plt.savefig('./plots/pies/subject' + str(int(dataset)) + '_score' + str(int(total_score)) + '.png')
        plt.cla()
        plt.clf()