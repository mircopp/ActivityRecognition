import numpy as np

class DataProcessing():

    def __init__(self, categorial_map, score_map, strategy='kubic'):
        self.categorial_map = categorial_map
        self.score_map = score_map
        self.ranking_strategy = strategy

    def fit(self, categories, scores):
        categories = np.unique(categories)
        self.categorial_map = None
        for i in range(categories):
            self.categorial_map[i] = categories[i]
            self.score_map[i] = scores[i]
        return self

    def get_activitiy(self, activity_number):
        return self.categorial_map[activity_number]

    def get_score(self, activity_number):
        if self.ranking_strategy == 'kubic':
            return self.score_map[activity_number] ** 3
        elif self.ranking_strategy == 'quadratic':
            return self.score_map[activity_number] **2
        #TODO add exponetial
        else:
            return self.score_map[activity_number]


DICTIONARY = {
    1   :   "Standing Still",
    2   :   "Sitting and relaxing",
    3   :   "Lying Down",
    4   :   "Walking",
    5   :   "Climbing stairs",
    6   :   "Waist bends forward",
    7   :   "Frontal elevation of arms",
    8   :   "Knees bending",
    9   :   "Cycling",
    10  :   "Jogging",
    11  :   "Running",
    12  :   "Jump front & back"
}

SCOREMAP = {
    1   :   2,
    2   :   1,
    3   :   0,
    4   :   3,
    5   :   4,
    6   :   4,
    7   :   4,
    8   :   4,
    9   :   5,
    10  :   6,
    11  :   7,
    12  :   4
}

def _split(arr, cond):
  return [arr[cond], arr[~cond]]

def _indexes_condition(arr, condition):
    return arr/condition == 1

def get_activities(activity_number):
    return DICTIONARY[activity_number]

def get_score(activity_number):
    return SCOREMAP[activity_number]**3

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

def sequentialize_vectors_non_labelled(data, sequence_length):
    clusters = []
    for i in np.unique(data[:, -1]):
        if i == 0:
            split_data = _split(data, data[:, -1] == i)[0]
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

def transform_tab_sep_to_csv(path, filename):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    data = []
    for x in content:
        tmp = x.split(sep='\t')
        data.append(tmp)
    data = np.array(data)
    X = data.astype(np.float64)
    np.savetxt(filename, X, delimiter=",")





