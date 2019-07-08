
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

#%%
# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

#%%
# Feature Extraction with RFE(recursive feature extraction)
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: ", fit.n_features_) 
print("Selected Features: ", fit.support_) 
print("Feature Ranking: ", fit.ranking_) 


#%%
# feature extraction with ExtraTrees
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)


#%%
# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])

#%%
# Feature Extraction with PCA
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: ", fit.explained_variance_ratio_) 
print(fit.components_)