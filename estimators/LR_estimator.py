from sklearn.linear_model import LogisticRegression
import numpy as np


def sigmoid(z):
  return (1/ (1+np.exp(-z)))

def gradient_db(x, y, w, b):

  w = np.squeeze(np.asarray(w))
  db = y - sigmoid(np.dot(w.T, x) + b)

  return db

class LR_u(LogisticRegression):
   
   def update_intercept(self, X, y, lr, epochs):
       
       w, b = self.coef_, self.intercept_
       N = len(X)

       for i in range(0, epochs):
        for j in range(N):            
            grad_db = gradient_db(X[j], y[j], w, b)
            b = b + (lr * grad_db)

       self.intercept_ = b     
 
       return self