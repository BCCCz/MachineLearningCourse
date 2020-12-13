from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_data():
    iris = load_iris()
    data = iris.data
    result = iris.target
    return data, result

data,result = get_data()

data_train, data_test, result_train, result_test = train_test_split(data, result, test_size=0.25)

KNN = KNeighborsClassifier()
KNN.fit(data_train,result_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')

predict = KNN.predict(data_test)
print(predict)
print(result_test)
print(accuracy_score(result_test,predict))

