'''
fits a linear model manually and with sklearn
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(123)
plt.style.use('ggplot')

def make_data(m=5, b=2, n=100):

        x = np.random.uniform(0,10,n)
        y = m*x + b
        u = np.random.normal(0, 5, n)
        y += np.random.rand(n)*u

        return x, y, u


def least_squares(x, y, u):

        n = len(x)
        A = np.c_[np.ones(n), x]
        
        # error matrix
        C = np.eye(n) * u**2

        # intercept, slope
        b, m = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y)

        # covariance matrix
        covar = np.linalg.inv(A.T.dot(np.linalg.solve(C, A)))

        b_u, m_u = 1.96 * np.sqrt(covar[0,0]), 1.96 * np.sqrt(covar[1,1])

        return b, m, b_u, m_u


# an example
m, b = 5, 3
x, y, u = make_data(m, b)

# manually
b_ls, m_ls, b_u, m_u = least_squares(x, y, u)
print( f''' Least Squares
                slope: {m_ls:0.3f} ± {m_u:0.3f} (True = {m})
                inter: {b_ls:0.3f} ± {b_u:0.3f} (True = {b})
        ''')

plt.errorbar(x, y, u, fmt='.')
plt.plot(x, m_ls*x + b_ls)

# sklearn
lin = LinearRegression()
lin.fit(x.reshape(-1, 1), y.reshape(-1, 1))
b_sk, m_sk = lin.intercept_[0], lin.coef_[0][0]
print( f''' Least Squares
                slope: {m_sk:0.3f} (True = {m})
                inter: {b_sk:0.3f} (True = {b})
        ''')

plt.plot(x, m_sk*x + b_sk, 'k--')
plt.show()
