import numpy as np
from sklearn.model_selection import train_test_split


def _split(arr, cond):
  return [arr[cond], arr[~cond]]

def _indexes_condition(arr, condition):
    return arr/condition == 1


def calculate_vector_length(vector_matrix):
    return np.linalg.norm(vector_matrix, axis=1)

def sequentialize_vectors(data, sequence_length):
    clusters = []
    for i in np.unique(data[:, -1]):
        if not i == 0:
            split_data = _split(data, data[:, -1] == i)[0]
            res = []
            for i in range(0, len(split_data), sequence_length):
                tmp = split_data[i:(i+sequence_length)]
                if len(tmp)<sequence_length:
                    missing_rows = sequence_length - len(tmp)
                    means = [np.mean(tmp, axis=0) for i in range(missing_rows)]
                    # zeros = np.zeros((missing_rows, np.shape(tmp)[1]))
                    tmp = np.vstack((tmp, means))
                tmp_res = []
                for x in tmp[:, :-1]:
                    tmp_res.extend(x)
                tmp_res = np.append(tmp_res, tmp[0, -1])
                res.extend([tmp_res])
            clusters.extend(np.array(res))
    return np.array(clusters)

def preprocess_data(data):
    data = np.column_stack((
        calculate_vector_length(data[:, :3]),
        data[:, 3:5],
        calculate_vector_length(data[:, 5:8]),
        calculate_vector_length(data[:, 8:11]),
        calculate_vector_length(data[:, 11:14]),
        calculate_vector_length(data[:, 14:17]),
        calculate_vector_length(data[:, 17:20]),
        calculate_vector_length(data[:, 20:23]),
        data[:, 23]
    ))
    clusters = []
    for i in np.unique(data[:, -1]):
        if not i == 0:
            split_data = _split(data, data[:, -1] == i)[0]
            mean_vals = np.mean(split_data, axis=0)
            variances = np.var(split_data, axis=0)
            standard_deviations = np.std(split_data, axis=0)
            max_vals = np.max(split_data, axis=0)
            min_vals = np.min(split_data, axis=0)
            percentiles = []
            for i in range(4):
                percentiles = np.append(percentiles, np.percentile(split_data, (i+1)*25))
            res = np.append(np.append(np.append(np.append(np.append(min_vals, max_vals), standard_deviations), variances), percentiles), mean_vals)
            clusters.append(res)
    return clusters

def train_split(X, y, train_size=0.5):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    matrix = np.column_stack((X,y))
    for i in np.unique(y):
        split_matrix = _split(matrix, matrix[:, -1] == i)[0]
        X_split = split_matrix[:, :-1]
        y_split = split_matrix[:, -1]
        X_split_train, X_split_test, y_split_train, y_split_test = train_test_split(X_split, y_split, train_size=train_size)
        X_train.extend(X_split_train)
        X_test.extend(X_split_test)
        y_train.extend(y_split_train)
        y_test.extend(y_split_test)
    return (np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test))






