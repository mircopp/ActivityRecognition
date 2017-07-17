import numpy as np
import pandas as ps

from seqmod.preprocessing import Sequentializer
from seqmod.model import SequentialSensoryDataModel
from seqmod.metrics import ScoreMap

if __name__ == '__main__':
    model_paths = ['sequential_sensory_data_model.bin']
    for path in model_paths:
        model = SequentialSensoryDataModel(path=path)
        models = model.get_models()

        dataset = 10

        CSV_FILES = ['csv/mHealth_subject' + str(int(dataset)) + '.csv']

        # Load data
        matrix = np.array(list(filter(lambda x: x[-1] != 0, Sequentializer(CSV_FILES).transform())))

        X, y = matrix[:, :-1], matrix[:, -1]

        X = model.normalize(X)

        for tmp_model in models:
            score = tmp_model['model'].score(X, y)
            print(tmp_model['description'], score)