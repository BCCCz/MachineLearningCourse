from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
import time

def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        fn(*args, **kwargs)
        print("%s cost %s second" % (fn.__name__, time.clock() - start))
    return _wrapper

def get_data():
    iris = load_iris()
    data = iris.data
    result = iris.target
    return data, result

@time_me
def Model():
    KNN = KNeighborsClassifier()
    KNN.fit(data_train,result_train)
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')

data,result = get_data()

data_train, data_test, result_train, result_test = train_test_split(data, result, test_size=0.25)

KNN = KNeighborsClassifier()
KNN.fit(data_train,result_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')

predict = KNN.predict(data_test)

Model()
print(predict)
print(result_test)
print(accuracy_score(result_test,predict))

