from pandas import read_csv
from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# split out validation dataset
arr = dataset.values
X = arr[:,0:4]
y = arr[:,4]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1)

# make predictions on validation dataset (svc)
model_svc = SVC(gamma='auto')
model_svc.fit(X_train, y_train)
predictions_svc = model_svc.predict(X_validation)

# make predictions on validation dataset (lda)
# unverified
model_lda = LinearDiscriminantAnalysis()
model_lda.fit(X_train, y_train)
predictions_lda = model_lda.predict(X_validation)

# evaluate predictions
print('SVC----------')
print(accuracy_score(y_validation, predictions_svc))
print(confusion_matrix(y_validation, predictions_svc))
print(classification_report(y_validation, predictions_svc))
print('LDA----------')
print(accuracy_score(y_validation, predictions_lda))
print(confusion_matrix(y_validation, predictions_lda))
print(classification_report(y_validation, predictions_lda))
