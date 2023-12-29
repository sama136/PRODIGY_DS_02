import numpy as np

import pandas as pd
df = pd.read_csv('train.csv')
print(df)
print(df.dtypes)
print(df.describe())
print(df.isna().sum())
print(df.dropna())
print(df.duplicated())
print(df.nunique())
print(df['Pclass'].unique())
print(df.head(15))
print(df['Name'])
print(df.groupby(['Sex', 'Survived'])['Survived'].count())
# count number of passenger in each class
print(df.groupby('Pclass')['PassengerId'].count().reset_index())

# relationship_analysis
# scatter matrix plot
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from pandas import read_csv
df = read_csv('train.csv')
df.plot(kind= 'scatter', x='Age', y='Fare')
plt.title('scatter plot of age vs fare')
plt.show()

# count plot with one variable
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
from pandas import read_csv
dataset = read_csv('train.csv')
sns.countplot(x='Survived', data=dataset, palette='hls')
plt.title('passenger count survived')
plt.show()

# histogram
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from pandas import read_csv
df = read_csv('train.csv')
Age_data = df['Age'].value_counts()
Age_data.plot.hist(bins=20, edgecolor='black', color='green')
plt.xlabel('Age')
plt.ylabel('count')
plt.title('age distribution')
plt.show()