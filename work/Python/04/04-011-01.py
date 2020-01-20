# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# %%
iris = load_iris()
print(iris.DESCR)

# %%
tmp_data = pd.DataFrame(iris.data, columns=iris.feature_names)
tmp_data['target'] = iris.target

# %%
data_iris = tmp_data[tmp_data['target']<=1]
print(data_iris.shape)
data_iris.head()

# %%
plt.scatter(data_iris.iloc[:, 0], data_iris.iloc[:, 1])

# %%
plt.scatter(data_iris.iloc[:, 0], data_iris.iloc[:, 1], c=data_iris['target'])

# %%
plt.scatter(data_iris.iloc[:, 2], data_iris.iloc[:, 3], c=data_iris['target'])

# %%
logit = LogisticRegression()
x_column_list = ['sepal length (cm)']
y_column_list = ['target']

x = data_iris[x_column_list]
y = data_iris[y_column_list]

logit.fit(x, y)

# %%
print(logit.coef_)
print(logit.intercept_)

# %%
