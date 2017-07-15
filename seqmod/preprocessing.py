import numpy as np

def _split(arr, cond):
  return [arr[cond], arr[~cond]]

def sequentialize_vectors(data, sequence_length):
    clusters = []
    for i in np.unique(data[:, -1]):
        if not i == 0:
            split_data = _split(data, data[:, -1] == i)[0]
            res = []
            for i in range(0, len(split_data), sequence_length):
                tmp = split_data[i:(i+sequence_length)]
                if len(tmp) < sequence_length:
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

def sequentialize_vectors_non_labelled(data, sequence_length):
    clusters = []
    split_data = _split(data, data[:, -1] == 0)[0]
    res = []
    for i in range(0, len(split_data), sequence_length):
        tmp = split_data[i:(i+sequence_length)]
        if len(tmp)<sequence_length:
            missing_rows = sequence_length - len(tmp)
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