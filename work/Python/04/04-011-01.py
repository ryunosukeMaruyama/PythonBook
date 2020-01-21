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
data_iris = tmp_data[tmp_data['target'] <= 1]
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
logit_multi = LogisticRegression()
x_column_list = ['sepal length (cm)', 'sepal width (cm)',
                 'petal length (cm)', 'petal width (cm)']
y_column_list = ['target']

x = data_iris[x_column_list]
y = data_iris[y_column_list]

logit_multi.fit(x, y)

print(logit_multi.coef_)
print(logit_multi.intercept_)

# %%
x_column_list = ['sepal width (cm)']
y_column_list = ['target']

X_train, X_test, y_train, y_test = train_test_split(
    data_iris[x_column_list], data_iris[y_column_list], test_size=0.2
)

logit2 = LogisticRegression()
logit2.fit(X_train, y_train)

print(logit2.coef_)
print(logit2.intercept_)

# %%
y_pred = logit2.predict(X_test)
y_pred

# %%
accuracy_score(y_test, y_pred)

# %%
x_column_list = ['sepal length (cm)', 'sepal width (cm)',
                 'petal length (cm)', 'petal width (cm)']
y_column_list = ['target']

X_train, X_test, y_train, y_test = train_test_split(
    data_iris[x_column_list], data_iris[y_column_list], test_size=0.2
)

logit_multi2 = LogisticRegression()
logit_multi2.fit(X_train, y_train)

print(logit_multi2.coef_)
print(logit_multi2.intercept_)

# %%
y_pred = logit_multi2.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
