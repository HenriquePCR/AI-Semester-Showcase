import pandas as ps
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
 
csv_data = ps.read_csv(r'C:\Users\rickz\Desktop\Henrique\CEFET\9semestre\LabIA\Iris_Data (1).csv')
csv_data.tail()
species_array = csv_data['species'].to_numpy()
d = csv_data.iloc[0:33, 4].values
d = np.append(d,csv_data.iloc[50:83, 4].values)
d = np.append(d,csv_data.iloc[100:133, 4].values)
v = []

for label in d:
    if label == 'Iris-setosa':
        v.append([1, 0, 0])
    elif label == 'Iris-versicolor':
        v.append([0, 1, 0])
    elif label == 'Iris-virginica':
        v.append([0, 0, 1])

y_encoded = np.array(v)

csv_data2 = ps.read_csv(r'C:\Users\rickz\Desktop\Henrique\CEFET\9semestre\LabIA\Iris_Data (1).csv')
csv_data2.tail()

x = csv_data2.iloc[0:33, 0:4].values

x2 = csv_data2.iloc[50:83, 0:4].values

x3 = csv_data2.iloc[100:133, 0:4].values

X = np.vstack((x, x2, x3))





# x = d.iloc[0:151, [0,2]].values
# def Perceptron(max_it, alfa):
#     W = np.zeros((3,X.shape[0]))
#     print(W.shape)
#     bias = [0,0]
#     t=0
#     e=1
#     while(t < max_it and e>0):
#         e = 0
#         for i in range(0,2):
#             transposed_matrix = np.transpose(X)
#             u = np.dot(W, transposed_matrix[i])+bias
#             print(u)
bias = np.zeros((3, 1))
W = np.zeros((3,X.shape[1]))
def Perceptron(max_it, alfa, W, bias):
    
    t=0
    e=1
    for _ in range(max_it):
        error = 0
        for xi, target in zip(X, y_encoded):
            u = np.dot(W, xi.reshape(-1, 1)) + bias
            y = np.where(u >= 0.0, 1, -1)
            e = target.reshape(-1, 1) - y
            # print(xi.reshape(1, -1))
            # print(xi)
            # print('e',e)
            W += 0.1 * np.dot(e, xi.reshape(1, -1))
            # print(w)
            bias += e * 0.1
            # print(bias)
            error += np.sum(e)
            # print(error)
            # print(error)

print(X)
Perceptron(1,0.1, W, bias)            


csv_data = ps.read_csv(r'C:\Users\rickz\Desktop\Henrique\CEFET\9semestre\LabIA\Iris_Data (1).csv')
csv_data.tail()
species_array = csv_data['species'].to_numpy()
d = csv_data.iloc[33:50, 4].values
d = np.append(d,csv_data.iloc[83:100, 4].values)
d = np.append(d,csv_data.iloc[133:150, 4].values)
v = []

for label in d:
    if label == 'Iris-setosa':
        v.append([1, 0, 0])
    elif label == 'Iris-versicolor':
        v.append([0, 1, 0])
    elif label == 'Iris-virginica':
        v.append([0, 0, 1])

y_test = np.array(v)

csv_data2 = ps.read_csv(r'C:\Users\rickz\Desktop\Henrique\CEFET\9semestre\LabIA\perceptron\Iris_Data (1).csv')


csv_data2.tail()

x = csv_data2.iloc[33:50, 0:4].values

x2 = csv_data2.iloc[83:100, 0:4].values

x3 = csv_data2.iloc[133:150, 0:4].values

x_test = np.vstack((x, x2, x3))

def score( x, y):
        misclassified_data_count = 0
        for xi, target in zip(x, y):
            u = np.dot(W, xi.reshape(-1, 1)) + bias
            y = np.where(u >= 0.0, 1, 0)
            output = y
            if(target != output ).any():
                misclassified_data_count += 1
        total_data_count = len(x)
        score_ = (total_data_count - misclassified_data_count)/total_data_count
        return score_

print('Prediction: %.3f' % score(x_test, y_test))
print('Result: %.3f' % score(X, y_encoded))