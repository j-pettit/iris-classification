from pathlib import Path

from pandas import read_csv
from pandas.plotting import scatter_matrix

from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# create figure path
Path("./fig").mkdir(parents=True, exist_ok=True)

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# summarize dataset
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('class').size())

# plot data
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.savefig('./fig/data_box_whisker.png')
dataset.hist()
pyplot.savefig('./fig/data_histogram.png')
scatter_matrix(dataset)
pyplot.savefig('./fig/data_scatter_matrix.png')

# split out validation dataset
arr = dataset.values
X = arr[:,0:4]
y = arr[:,4]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1)

# spot check and evaluate algorithms
models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto')),
]
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean()} ({cv_results.std()})')

# Plot algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithms Comparison')
pyplot.savefig('./fig/algorithms_comparison.png')
