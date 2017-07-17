import numpy as np
import pandas as ps


class Sequentializer():

    def __init__(self, files, sequence_length=50):
        self.filenames = files
        self.sequence_length = sequence_length

    def get_files(self):
        return self.filenames

    def get_sequence_length(self):
        return self.sequence_length

    def transform(self):
        # Load the training data
        data_stack = np.array([])
        for file in self.filenames:
            data = ps.read_csv(file, sep=',').as_matrix()
            data = self._sequentialize_vectors(data)
            data_stack = np.vstack((data_stack, data)) if data_stack.any() else np.vstack(data)
        return data_stack

    def _split(self, arr, cond):
        return [arr[cond], arr[~cond]]

    def _sequentialize_vectors(self, data):
        clusters = []
        for i in np.unique(data[:, -1]):
            split_data = self._split(data, data[:, -1] == i)[0]
            res = []
            for i in range(0, len(split_data),  self.sequence_length):
                tmp = split_data[i:(i+ self.sequence_length)]
                if len(tmp) <  self.sequence_length:
                    missing_rows =  self.sequence_length - len(tmp)
                    means = [np.mean(tmp, axis=0) for i in range(missing_rows)]
                    tmp = np.vstack((tmp, means))
                tmp_res = []
                for x in tmp[:, :-1]:
                    tmp_res.extend(x)
                tmp_res = np.append(tmp_res, tmp[0, -1])
                res.extend([tmp_res])
            clusters.extend(np.array(res))
        return np.array(clusters)

def transform_to_csv(path, filename, sep='\t'):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    data = []
    for x in content:
        tmp = x.split(sep=sep)
        data.append(tmp)
    data = np.array(data)
    X = data.astype(np.float64)
    np.savetxt(filename, X, delimiter=",")