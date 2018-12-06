 # import libraries that needed
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix
import pickle

#import a data of breast cancer
from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
#the information present in breast cancer data
print(cancer_data)

#keys in cancer data
cancer_data.keys()
print(cancer_data['data'])
print(cancer_data['DESCR'])
print(cancer_data['target_names'])
print(cancer_data['target'])
print(cancer_data['feature_names'])

#converting above data to a features column
cancer_dataset = pd.DataFrame(np.c_[cancer_data['data'], cancer_data['target']], columns = np.append(cancer_data['feature_names'], ['target']))
#first rows view of cancer data
cancer_dataset.head()

print("counter plot")
#count plot for maligent and bengnign
sns.countplot(cancer_dataset['target'], label = "Count") 
print("scatterplot")
#scatter plot between mean are and mean smoothness of tumour
plt.figure(figsize=(10,10))
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = cancer_dataset)

print("heatmap")
plt.figure(figsize=(10,10)) 
sns.heatmap(cancer_dataset.corr(), annot=True) 

# Let's drop the target label coloumns
X = cancer_dataset.drop(['target'],axis=1)
y = cancer_dataset['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#model
#Support Vector Machine classifier
svc_model = SVC()
#fitting of model
svc_model.fit(X_train, y_train)
#prediction of model
prediction_y = svc_model.predict(X_test)
cm = confusion_matrix(y_test, prediction_y)
plt.figure(figsize=(10,10)) 
print("heatmap")
sns.heatmap(cm, annot=True)

print(classification_report(y_test, prediction_y))
#saving of model to disk using pickle 
filename_1 = 'svm_model.sav'
pickle.dump(svc_model, open(filename_1, 'wb'))



#normalization of train
min_train = X_train.min()
print(min_train)
trange = (X_train - min_train).max()
print(trange)
X_train_normal = (X_train - min_train)/trange
print(X_train_normal)

#scatterplot
print("scatterplot")
plt.figure(figsize=(5,5)) 
sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)
print("scatterplot")
plt.figure(figsize=(5,5)) 
sns.scatterplot(x = X_train_normal['mean area'], y = X_train_normal['mean smoothness'], hue = y_train)

#normalization of test
min_test = X_test.min()
test_range = (X_test - min_test).max()
X_test_normal = (X_test - min_test)/test_range


#svc model
svc_model = SVC()
svc_model.fit(X_train_normal, y_train)
y_predict = svc_model.predict(X_test_normal)
cm = confusion_matrix(y_test, y_predict)

#saving of model to disk using pickle 
filename_2 = 'svm_normalization.sav'
pickle.dump(svc_model, open(filename_2, 'wb'))

print("heatmap")
#confusion matrix representation
sns.heatmap(cm,annot=True,fmt="d")
print(classification_report(y_test,y_predict))
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


#by adjusting c and gamma parameter of SVC
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_normal,y_train)
print(grid.best_params_)
print(grid.best_estimator_)

#final prediction
grid_predictions = grid.predict(X_test_normal)

print("heatmap")
#confusion matrix
plt.figure(figsize=(5,5)) 
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)

#classification report gives precision recall etc
print(classification_report(y_test,grid_predictions))

#saving of model to disk using pickle 
filename_3 = 'svm_gridsearch.sav'
pickle.dump(grid, open(filename_3, 'wb'))



# load the model from disk svc
loaded_model_1= pickle.load(open(filename_1, 'rb'))
result_svm = loaded_model_1.score(X_test, y_test)
print("Score result of SVC : ",result_svm)

# load the model from disk after normalization
loaded_model_2 = pickle.load(open(filename_2, 'rb'))
result_normalization = loaded_model_2.score(X_test_normal, y_test)
print("Score result after notmalization :",result_normalization)

# load the model from disk after adjusting gamma and c parameter
loaded_model_3 = pickle.load(open(filename_3, 'rb'))
result_grid = loaded_model_3.score(X_test_normal, y_test)
print("Score result after adjusting c and gamma parameter of SVC :",result_grid)