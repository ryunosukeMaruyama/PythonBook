# %%
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from sklearn.datasets import load_iris
import pandas as pd

# %%
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.loc[df['target']==0, 'target'] = 'setosa'
df.loc[df['target']==1, 'target'] = 'versicolor'
df.loc[df['target']==2, 'target'] = 'virginica'

# %%
df.head()
# %%
plt.title('iris data plot')
plt.xlabel('x axis name')
plt.ylabel('y axis name')
plt.hist(df['sepal length (cm)'])
plt.show()

sns.distplot(df['sepal length (cm)'])

# %%
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
plt.show()

sns.jointplot('sepal length (cm)', 'petal length (cm)', data=df)

# %%
x = [1, 2, 3, 4]
y = [2, 5, 7, 9]

plt.plot(x, y)
plt.show()

sns.lineplot(x, y)

# %%
sns.boxplot(x='target', y='sepal length (cm)', data=df)

# %%
corr = df.corr()
sns.heatmap(corr)

# %%
corr

# %%
sns.pairplot(df, hue='target')

# %%
