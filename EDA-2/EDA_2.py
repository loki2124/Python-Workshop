
from matplotlib import pyplot
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
import numpy


#%%
# Correction Matrix Plot (generic)

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
correlations = data.corr()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()


#%%
# Box and Whisker Plots
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()


#%%
#Univariate histograms
data.hist()
pyplot.show()


#%%
#Univariate density plots
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()

#%%
#Scatterplot matrix
scatter_matrix(data)
pyplot.show()
