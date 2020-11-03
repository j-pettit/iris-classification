from pandas import read_csv
from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# summarize dataset
# print(dataset.shape)
# print(dataset.head(20))
# print(dataset.describe())
# print(dataset.groupby('class').size())

# plot data
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# dataset.hist()
# scatter_matrix(dataset)
# pyplot.show()

# split out validation dataset
arr = dataset.values
X = arr[:,0:4]
y = arr[:,4]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1)

# spot check and evaluate algorithms
# models = [
#     ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
#     ('LDA', LinearDiscriminantAnalysis()),
#     ('KNN', KNeighborsClassifier()),
#     ('CART', DecisionTreeClassifier()),
#     ('NB', GaussianNB()),
#     ('SVM', SVC(gamma='auto')),
# ]
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print(f'{name}: {cv_results.mean()} ({cv_results.std()})')

# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithms Comparison')
# pyplot.show()

# make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
