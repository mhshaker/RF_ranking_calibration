import Data.data_provider as dp
import numpy as np

x = np.array([1,0,0,0,0,1])

y = np.where(x==1)

print("y", y)