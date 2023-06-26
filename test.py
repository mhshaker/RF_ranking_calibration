import numpy as np
import math
import time
from joblib import Parallel, delayed

def func(x,y):
    return math.factorial(x) + y

res = [1]
res = Parallel(n_jobs=-1)(delayed(func)(x,y) for x,y in zip(range(10000), range(10000)))
print("res", len(res))