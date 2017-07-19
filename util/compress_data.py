import numpy as np
import pandas as ps
import os

def transform(datasets):
    for i in datasets:
        data = ps.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'csv/mHealth_subject' + str(int(i)) + '.csv')).as_matrix()
        non_labelled = np.array(list(filter(lambda x: int(x[-1]) == 0, data)))[:, :-1]
        labelled = np.array(list(filter(lambda x: int(x[-1]) != 0, data)))
        np.savetxt(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'data_collection/non_labelled/mHealth_non_labelled_subject' + str(int(i)) + '.csv'), non_labelled, delimiter=",")
        np.savetxt(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'data_collection/labelled/mHealth_subject' + str(int(i)) + '.csv'), labelled, delimiter=",")


if __name__ == '__main__':
    transform(np.linspace(1, 10, 10))
