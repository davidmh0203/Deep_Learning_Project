
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from common.functions import sigmoid, softmax


def get_data():
    iris = load_iris()
    features = iris["data"]
    feature_names = iris['feature_names']

    df = pd.DataFrame(features, columns=feature_names)
    df['target'] = iris['target']

    x = df[feature_names]
    y = df['target']


    return train_test_split(x, y, stratify=y, test_size=0.3, random_state=100)


def init_network():
    W1 = np.random.randn(4, 10)
    b1 = np.zeros(10)
    W2 = np.random.randn(10, 10)
    b2 = np.zeros(10)
    W3 = np.random.randn(10, 3)
    b3 = np.zeros(3)

    return {'W1': W1, 'W2': W2, 'W3': W3, 'b1': b1, 'b2': b2, 'b3': b3}


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y



iris_target_names = ["setosa", "versicolor", "virginica"]



x_train, x_test, t_train, t_test = get_data()

network = init_network()
accuracy_cnt = 0
for i in range(len(x_test)):
    y = predict(network, x_test.values[i])
    p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스를 얻는다.
    predicted_species = iris_target_names[p]  # 예측된 품종 이름
    actual_species = iris_target_names[t_test.values[i]]  # 실제 품종 이름

    if p == t_test.values[i]:  #t는 정답, 티쳐값
        accuracy_cnt += 1  #맞췄다


    print(f"Predicted: {predicted_species}, Actual: {actual_species}")

print("Accuracy: " + str(float(accuracy_cnt) / len(x_test)))
