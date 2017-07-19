import numpy as np
import pandas as ps


class Sequentializer():

    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length

    def get_sequence_length(self):
        return self.sequence_length

    def transform(self, X, y):
        if y.any():
            XY = self._sequentialize_vectors(np.column_stack((X, y)))
            return XY[:, :-1], XY[:, -1]
        else:
            return self._sequentialize_vectors_non_labelled(X)

    def _split(self, arr, cond):
        return [arr[cond], arr[~cond]]

    def _sequentialize_vectors(self, XY):
        XY_sequentialized = []
        for i in np.unique(XY[:, -1]):
            XY_split = self._split(XY, XY[:, -1] == i)[0]
            for i in range(0, len(XY_split),  self.sequence_length):
                XY_curr_sequence = XY_split[i:(i+ self.sequence_length)]
                if len(XY_curr_sequence) <  self.sequence_length:
                    missing_rows =  self.sequence_length - len(XY_curr_sequence)
                    means = [np.mean(XY_curr_sequence, axis=0) for i in range(missing_rows)]
                    XY_curr_sequence = np.append(XY_curr_sequence, means, axis=0)
                y_sequentialized = []
                for x in XY_curr_sequence[:, :-1]:
                    y_sequentialized.extend(x)
                y_sequentialized = np.append(y_sequentialized, XY_curr_sequence[0, -1])
                XY_sequentialized.append(y_sequentialized)
        return np.array(XY_sequentialized)

    def _sequentialize_vectors_non_labelled(self, XY):
        XY_sequentialized = []
        for i in range(0, len(XY), self.sequence_length):
            XY_curr_sequence = XY[i:(i + self.sequence_length)]
            if len(XY_curr_sequence) <  self.sequence_length:
                missing_rows =  self.sequence_length - len(XY_curr_sequence)
                means = [np.mean(XY_curr_sequence, axis=0) for i in range(missing_rows)]
                XY_curr_sequence = np.vstack((XY_curr_sequence, means))
            y_sequentialized = []
            for x in XY_curr_sequence[:, :-1]:
                y_sequentialized.extend(x)
            XY_sequentialized.append(y_sequentialized)
        return np.array(XY_sequentialized)