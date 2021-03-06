# %%
import re
import json
import requests
import random
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.DESCR)

# %%
data_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
data_boston['PRICE'] = boston.target

data_boston.head()

# %%
sns.jointplot('RM', 'PRICE', data=data_boston)

# %%
sns.pairplot(data_boston, vars=['PRICE', 'RM', 'DIS'])

# %%
lr = LinearRegression()

x_column_list = ['RM']
y_column_list = ['PRICE']

data_boston_x = data_boston[x_column_list]
data_boston_y = data_boston[y_column_list]

lr.fit(data_boston_x, data_boston_y)

print(lr.coef_)
print(lr.intercept_)

# %%
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

data_boston_x = data_boston[x_column_list_for_multi]
data_boston_y = data_boston[y_column_list_for_multi]

lr_multi.fit(data_boston_x, data_boston_y)

# %%
print(lr_multi.coef_)
print(lr_multi.intercept_)

# %%

X_train, X_test, Y_train, Y_test = train_test_split(
    data_boston_x, data_boston_y, test_size=0.3, random_state=123)

# %%
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# %%
lr_multi2 = LinearRegression()
lr_multi2.fit(X_train, Y_train)
print(lr_multi2.coef_)
print(lr_multi2.intercept_)

# %%
y_pred = lr_multi2.predict(X_test)
print(y_pred)

# %%
print(y_pred - Y_test)


# %%

x_column_list = ['RM']
y_column_list = ['PRICE']
X_train, X_test, y_train, y_test = train_test_split(
    data_boston[x_column_list],
    data_boston[y_column_list],
    test_size=0.3,
    random_state=123
)

lr_single = LinearRegression()

lr_single.fit(X_train, y_train)
y_pred = lr_single.predict(X_test)

print(mean_absolute_error(y_pred, y_test))

# %%
X_train, X_test, y_train, y_test = train_test_split(
    data_boston[x_column_list_for_multi],
    data_boston[y_column_list_for_multi],
    test_size=0.3,
    random_state=123
)
lr_multi2 = LinearRegression()
lr_multi2.fit(X_train, y_train)
y_pred = lr_multi2.predict(X_test)

print(mean_absolute_error(y_pred, y_test))

# %%


url_path = 'https://www.land.mlit.go.jp/webland/api/TradeListSearch?from=20142&to=20153&area=10'
request_result = requests.get(url_path)
data_json = request_result.json()['data']
print(len(data_json))

# %%
data_json[0]

# %%
data_pd = pd.io.json.json_normalize(data_json)
data_pd.shape

# %%
data_pd.head()

# %%
data_from_csv = pd.read_csv('13_Tokyo_20171_20184.csv', encoding='cp932')
data_from_csv.shape

# %%
print(data_from_csv.iloc[0])

# %%
data_from_csv.head(10)

# %%
data_from_csv.columns

# %%
data_from_csv['種類'].unique()

# %%
data_used_apartment = data_from_csv.query('種類=="中古マンション等"')
data_used_apartment.shape

# %%
data_used_apartment.isnull().sum()

# %%
columns_name_list = [
    '最寄駅：距離（分）',
    '間取り',
    '面積（㎡）',
    '建築年',
    '建物の構造',
    '建ぺい率（％）',
    '容積率（％）',
    '市区町村名',
    '取引価格（総額）'
]
data_selected = data_used_apartment[columns_name_list]
print(data_selected.shape)

# %%
data_selected_dropna = data_selected.dropna(how='any')
print(data_selected_dropna.shape)
data_selected_dropna.iloc[0]

# %%
data_selected_dropna['建築年'].unique()

# %%
data_selected_dropna = data_selected_dropna[data_selected_dropna['建築年'].str.match(
    '^平成|昭和')]
wareki_to_seireki = {'昭和': 1926-1, '平成': 1989-1}

building_year_list = data_selected_dropna['建築年']

building_age_list = []
for building_year in building_year_list:
    building_year_split = re.search(r'(.+?)([0-9]+|元)年', building_year)
    seireki = wareki_to_seireki[building_year_split.groups()[0]] + \
        int(building_year_split.groups()[1])
    building_age = 2019 - seireki
    building_age_list.append(building_age)

data_selected_dropna['築年数'] = building_age_list
data_added_building_age = data_selected_dropna.drop('建築年', axis=1)
data_added_building_age.head()

# %%
# ダミー変数化しないものリスト
columns_name_list = [
    '最寄駅：距離（分）',
    '面積（㎡）',
    '築年数',
    '建ぺい率（％）',
    '容積率（％）',
    '取引価格（総額）'
]

# ダミー変数リスト
dummy_list = [
    '間取り',
    '建物の構造',
    '市区町村名'
]

# ダミー変数を追加
data_added_dummies = pd.concat(
    [
        data_added_building_age[columns_name_list],
        pd.get_dummies(data_added_building_age[dummy_list],
                       drop_first=True)
    ],
    axis=1
)

# %%
print(data_added_dummies.dtypes)

# %%
data_added_dummies['面積（㎡）'] = data_added_dummies['面積（㎡）'].astype(float)
data_added_dummies = data_added_dummies[~data_added_dummies['最寄駅：距離（分）'].str.contains('\?')]
data_added_dummies['最寄駅：距離（分）'] = data_added_dummies['最寄駅：距離（分）'].astype(float)

# %%
plt.hist(data_added_dummies['取引価格（総額）'])
plt.show()

# %%
tmp_data = data_added_dummies[data_added_dummies['取引価格（総額）'] < 60000000]
print(tmp_data.shape)
plt.hist(tmp_data['取引価格（総額）'])
plt.show

# %%
data_added_dummies = data_added_dummies[data_added_dummies['取引価格（総額）'] < 60000000]

# %%
lr = LinearRegression()

x_column_list = ['面積（㎡）']
y_column_list = ['取引価格（総額）']

x = data_added_dummies[x_column_list]
y = data_added_dummies[y_column_list]

lr.fit(x, y)

print(lr.coef_)
print(lr.intercept_)

# %%
x = data_added_dummies.drop('取引価格（総額）', axis=1)
y = data_added_dummies['取引価格（総額）']

print(x.head())
print(y.head())

# %%
lr_multi = LinearRegression()
lr_multi = lr_multi.fit(x, y)

print(lr_multi.coef_)
print(lr_multi.intercept_)

# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%
lr_multi2 = LinearRegression()
lr_multi2.fit(X_train, y_train)
print(lr_multi2.coef_)
print(lr_multi2.intercept_)

# %%
y_pred = lr_multi2.predict(X_test)
print(y_pred - y_test)

# %%
from sklearn.metrics import mean_absolute_error

x_column_list = ['面積（㎡）']
y_column_list = ['取引価格（総額）']

X_train, X_test, y_train, y_test = train_test_split(
    data_added_dummies[x_column_list],
    data_added_dummies[y_column_list],
    test_size=0.3
)

lr_single = LinearRegression()
lr_single.fit(X_train, y_train)
y_pred = lr_single.predict(X_test)

mean_absolute_error(y_pred, y_test)

# %%
x = data_added_dummies.drop('取引価格（総額）', axis=1)
y = data_added_dummies['取引価格（総額）']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

lr_multi2 = LinearRegression()
lr_multi2.fit(X_train, y_train)
y_pred = lr_multi2.predict(X_test)

mean_absolute_error(y_pred, y_test)

# %%
