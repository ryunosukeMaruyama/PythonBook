# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# %%
boston = load_boston()
data_boston = pd.DataFrame(boston.data, columns=boston.feature_names)

# %%
data_boston['PRICE'] = boston.target

lr_multi = LinearRegression()
x_column_list_for_multi = [
    'CRIM',
    'ZN',
    'INDUS',
    'CHAS',
    'NOX',
    'RM',
    'AGE',
    'DIS',
    'RAD',
    'TAX',
    'PTRATIO',
    'B',
    'LSTAT'
]
y_column_list_for_multi = ['PRICE']

lr_multi.fit(data_boston[x_column_list_for_multi],
             data_boston[y_column_list_for_multi])

print(lr_multi.coef_)
print(lr_multi.intercept_)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    data_boston[x_column_list_for_multi],
    data_boston[y_column_list_for_multi],
    test_size=0.3
)
lr_multi2 = LinearRegression()

lr_multi2.fit(X_train, y_train)
print(lr_multi2.coef_)
print(lr_multi2.intercept_)

y_pred = lr_multi2.predict(X_test)

print(mean_absolute_error(y_pred, y_test))

# %%
# ラッソ回帰
lasso = Lasso(alpha=0.01, normalize=True)
lasso.fit(X_train, y_train)
print(lasso.coef_)
print(lasso.intercept_)

# %%
y_pred_lasso = lasso.predict(X_test)
mean_absolute_error(y_pred_lasso, y_test)

# %%
# リッジ回帰
ridge = Ridge(alpha=0.01, normalize=True)
ridge.fit(X_train, y_train)
print(ridge.coef_)
print(ridge.intercept_)

# %%
y_pred = ridge.predict(X_test)
mean_absolute_error(y_pred, y_test)

# %%
