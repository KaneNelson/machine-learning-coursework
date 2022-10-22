import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math

with open('CPI.txt') as f:
    lines = f.readlines()
names = []
data = []




for line in lines:
    names.append(line.strip().split(',')[0])
    data.append(np.array(line.strip().split(',')[1:]).astype(float))

russia = [67.62, 31.68, 10.00, 3.87, 12.90]

scaler = MinMaxScaler()
scaler.fit(data)
data_normalized = scaler.transform(data)
russia_normalized = [0.6099, 0.3754, 0.0948, 0.5658, 0.9058]




def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)

def CPI_kNN(dat, target, k):
    euclids = euclid_values(dat, target)
    idxsort = np.argsort(euclids)
    sum = 0.0
    for i in range(k):
        sum += data[idxsort[i]][5]
    return sum/k

def CPI_kNN_weighted(dat, target, k):
    euclids = euclid_values(dat, target)
    weights = get_weights(dat, target)
    cpixweights = [weights[i] * data[i][5] for i in range(len(weights))]
    idxsort = np.argsort(euclids)[::-1]

    weightsum = 0.0
    cpixweightsum = 0.0

    for i in range(k):
        weightsum += weights[idxsort[i]]
        cpixweightsum += cpixweights[idxsort[i]]
    return cpixweightsum/weightsum

def euclid_values(dat, target):
    return[euclidean_distance(dat[x], target) for x in range(len(dat))]

def get_weights(dat, target):
    euclids = euclid_values(dat, target)
    weights = [(x**-2) for x in euclids]
    return weights

def display_data(dat, target):
    euclids = euclid_values(dat, target)
    idxsort = np.argsort(euclids)
    print('Country'.ljust(25), 'Euclid Value'.ljust(15), 'CPI'.ljust(15))
    for i in idxsort:
        print(names[i].ljust(25), str(round(euclids[i], 4)).ljust(15), str(data[i][5]).ljust(15))

def display_data_weighted(dat, target):
    euclids = euclid_values(dat, target)
    weights = get_weights(dat, target)
    idxsort = np.argsort(weights)[::-1]
    print('Country'.ljust(25), 'Euclid Value'.ljust(15),'CPI'.ljust(15), 'Weight'.ljust(15), 'Weight * CPI'.ljust(15))

    for i in idxsort:
        print(names[i].ljust(25), str(round(euclids[i], 4)).ljust(15), str(data[i][5]).ljust(15),
              str(round(weights[i], 4)).ljust(15), str(round((weights[i] * data[i][5]), 4)).ljust(15))


# 1
display_data(data, russia)
print(f'CPI for 3-NN is : {round(CPI_kNN(data, russia, 3), 4)}')
print()

# 2
display_data_weighted(data, russia)
print(f'CPI for 16-NN is : {round(CPI_kNN_weighted(data, russia, 16), 4)}')
print()

# 3
display_data(data_normalized, russia_normalized)
print(f'CPI for 3-NN weighted is : {round(CPI_kNN(data_normalized, russia_normalized, 3), 4)}')
print()

# 4
display_data_weighted(data_normalized, russia_normalized)
print(f'CPI for 16-NN weighted is : {round(CPI_kNN_weighted(data_normalized, russia_normalized, 16), 4)}')



