import pickle as pc
import numpy as np
import pandas as ps

from preprocessing import compress_data

if __name__ == '__main__':

    sequence_length = 50
    model, sc = pc.load(open('model' + str(sequence_length) + '.bin', 'rb'))

    use_datasets = np.linspace(1, 2, 1)
    CSV_FILES = ['csv/mHealth_subject' + str(int(i)) + '.csv' for i in use_datasets]

    matrix = np.array([])
    # Load data
    for file in CSV_FILES:
        data = ps.read_csv(file, sep=',').as_matrix()
        data = compress_data.sequentialize_vectors(data, sequence_length=sequence_length)
        matrix = np.vstack((matrix, data)) if matrix.any() else np.vstack(data)

    for i in np.linspace(1, 12, 12):
        filtered = np.array(list(filter(lambda x: x[-1]==i, matrix)))
        filtered = sc.transform(filtered[:, :-1])
        prediction = model.predict(filtered)
        print(prediction)


    pass