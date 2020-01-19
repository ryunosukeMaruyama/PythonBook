# %%
from sklearn.datasets import load_iris
iris = load_iris()
iris.target

# %%
iris.data

# %%
import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.head()

# %%
df.info()

# %%
from sklearn.preprocessing import MinMaxScaler
X = iris.data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X[0:5]

# %%
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

accuracy_score(clf.predict(iris.data), iris.target)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

accuracy_score(clf.predict(X_test), y_test)

# %%
