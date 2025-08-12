from flamby.datasets.fed_tcga_brca import FedTcgaBrca
import numpy as np

def load_data(center=0):
    dataset = FedTcgaBrca(center=center)
    features = []
    labels = []
    for i in range(len(dataset)):
        x, y = dataset[i]  # y is a 2-element tensor
        features.append(x.numpy())
        labels.append(y.numpy())  # Keep both event and time
    return np.array(features), np.array(labels)
