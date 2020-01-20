# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import requests
import json
import re

data_from_csv = pd.read_csv('13_Tokyo_20171_20184.csv', encoding='cp932')
data_used_apartment = data_from_csv.query('種類=="中古マンション等"')

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
data_selected_dropna = data_selected.dropna(how='any')
data_selected_dropna = data_selected_dropna[data_selected_dropna['建築年'].str.match(
    '^平成|昭和')]

wareki_to_seireki = {'昭和': 1926 - 1, '平成': 1989 - 1}

# %%
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

# ダミー変数追加
data_added_dummies = pd.concat(
    [data_added_building_age[columns_name_list],
     pd.get_dummies(data_added_building_age[dummy_list],
                    drop_first=True)],
    axis=1
)

# 文字列を数値化
data_added_dummies['面積（㎡）'] = data_added_dummies['面積（㎡）'].astype(float)
data_added_dummies = data_added_dummies[~data_added_dummies['最寄駅：距離（分）'].str.contains(
    '\?')]
data_added_dummies['最寄駅：距離（分）'] = data_added_dummies['最寄駅：距離（分）'].astype(float)

# 6000万円以下のデータのみ抽出
data_added_dummies = data_added_dummies[data_added_dummies['取引価格（総額）'] < 60000000]

# %%
x = data_added_dummies.drop('取引価格（総額）', axis=1)
y = data_added_dummies['取引価格（総額）']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)
print(lr_multi.coef_)
print(lr_multi.intercept_)

y_pred = lr_multi.predict(X_test)

mean_absolute_error(y_pred, y_test)

# %%
lasso = Lasso(alpha=1, normalize=True)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

mean_absolute_error(y_pred_lasso, y_test)

# %%
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

mean_absolute_error(y_pred_ridge, y_test)

# %%
