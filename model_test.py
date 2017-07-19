import pandas as ps
from actreg.seqmod import SequentialSensoryDataModel

if __name__ == '__main__':
    model = SequentialSensoryDataModel()
    model.load_model()

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