import numpy as np

def transform_tab_sep_to_csv(path, filename):
    with open(path) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    data = []
    for x in content:
        tmp = x.split(sep='\t')
        data.append(tmp)
    data = np.array(data)
    X = data.astype(np.float64)
    np.savetxt(filename, X, delimiter=",")