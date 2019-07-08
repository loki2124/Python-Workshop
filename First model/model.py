from pandas import read_csv
from sklearn.model_selection import train_test_split,cross_val_score,KFold, ShuffleSplit, LeaveOneOut
from sklearn.linear_model import LogisticRegression



#%%
# Evaluate using a train and a test set
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print("Accuracy:", result*100.0 ) 


#%%
# Evaluate using Cross Validation
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy:", (results.mean()*100.0, results.std()*100.0)) 


#%%
# Evaluate using Shuffle Split Cross Validation
n_splits = 10
test_size = 0.33
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy:", (results.mean()*100.0, results.std()*100.0) )

#%%
# Evaluate using Leave One Out Cross Validation
num_folds = 10
loocv = LeaveOneOut()
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy:", (results.mean()*100.0, results.std()*100.0) )
