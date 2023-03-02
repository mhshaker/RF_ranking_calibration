import Data.data_provider as dp
import numpy as np
import matplotlib.pyplot as plt

data_list = ["spambase", "climate", "QSAR", "vertebral", "ionosphere"]

for data in data_list:
    
    X, y = dp.load_data(data)
    print(X.shape)