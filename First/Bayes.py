from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def get_data():
    iris = load_iris()
    data = iris.data
    result = iris.target
    return data,result

data,result = get_data()
Gauss = GaussianNB()
Gauss = Gauss.fit(data,result)
pred = Gauss.predict(data)

print(pred)
print(accuracy_score(result,pred))