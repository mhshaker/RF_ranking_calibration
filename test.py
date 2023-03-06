import Data.data_provider as dp
import numpy as np

x = np.array([1,0,0,0,0,1])
y = np.array([1,2,2,2,2,10])

for i ,j in zip(x,y):
    print(f"x {i} y {j}")