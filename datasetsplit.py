from sklearn.model_selection import train_test_split
import h5py
import numpy as np

data = h5py.File('total_dataset.h5', 'r')
x, y = data['x'], data['y']
print(x.shape)
print(y.shape)
x = np.array(x)
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x_train.shape)
print(y_train.shape)
trainfile = h5py.File('train_set.h5', 'w')
trainfile.create_dataset('x_train', data=x_train)
trainfile.create_dataset('y_train', data=y_train)
trainfile.close()

testfile = h5py.File('test_set.h5', 'w')
testfile.create_dataset('x_test', data=x_test)
testfile.create_dataset('y_test', data=y_test)
testfile.close()