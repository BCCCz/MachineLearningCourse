from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def get_data():
    iris = load_iris()
    data = iris.data
    result = iris.target
    return data,result

data,result = get_data()
data_train,data_test,result_train,result_test = train_test_split(data,result, test_size=0.3)

Gauss = GaussianNB()
Gauss = Gauss.fit(data_train,result_train)
pred = Gauss.predict(data_test)

print(pred)
print(accuracy_score(result_test,pred))