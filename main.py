import numpy as np

from train import train_model

if __name__ == '__main__':
    size = 50
    train_model(sequence_length=size, train_datasets=np.linspace(1, 1, 1))

