# View first 20 rows
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
peek = data.head(20)
print(peek)

# Dimensions of your data
#%%
shape = data.shape
print(shape)

#%%
# Data Types for Each Attribute
types = data.dtypes
print(types)

#%%
# Class Distribution
class_counts = data.groupby('class').size()
print(class_counts)

#%%
# Statistical Summary
description = data.describe()
print(description)

#%%
# Pairwise Pearson correlations
correlations = data.corr(method='pearson')
print(correlations)

#%%Skewness
skew = data.skew()
print(skew)
