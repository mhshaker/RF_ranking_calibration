from sklearn.linear_model import LogisticRegression

class LR_u(LogisticRegression):

    def update_intercept(self, X, y):
        # minize loss by keaping cof/beta constant and only changing alpha/intercept
        pass
