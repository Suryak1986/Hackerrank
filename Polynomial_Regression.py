# Enter your code here. Read input from STDIN. Print output to STDOUT
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Read the training data
F,N = map(int,input().split())
train = np.array([input().split() for _ in range(N)],float)
X_train = train[:,0:F]
y_train =train[:,-1]

# Read the test data
T = int(input())
X_test = np.array([input().split() for _ in range(T)],float)

# Model
pol = PolynomialFeatures(degree=3,include_bias=False)
model = LinearRegression()
model.fit(pol.fit_transform(X_train),y_train)

# Prediction
y_test = model.predict(pol.transform(X_test))

print('\n'.join(list(map(str,y_test))))
