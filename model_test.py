import numpy as np
import pandas as ps

from seqmod.preprocessing import Sequentializer
from seqmod.model import SequentialSensoryDataModel
from seqmod.metrics import ScoreMap

if __name__ == '__main__':
    model_paths = ['sequential_sensory_data_model_fifth_iteration.bin']
    for path in model_paths:
        model = SequentialSensoryDataModel(path=path)
        models = model.get_models()

        dataset = 10

        file = 'data_collection/labelled/mHealth_subject' + str(int(dataset)) + '.csv'

        # Load data
        XY = ps.read_csv(file, sep=',').as_matrix()
        X, y = model.sequentialize(XY[:, :-1], XY[:, -1])

        X = model.normalize(X)

        for tmp_model in models:
            score = tmp_model['model'].score(X, y)
            print(tmp_model['description'], score)