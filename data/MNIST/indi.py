import os
import gzip
import numpy as np
import matplotlib.pyplot as plt



dl_list = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]
dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

file_path1 = dataset_dir + '/' + dl_list[0]
data1 = gzip.open(file_path1,'rb')
data1= data1.read()
data1 = np.frombuffer(data1,np.uint8,offset=16)
train_img = data1.reshape(-1,784)
print(train_img.shape)

file_path2 = dataset_dir + '/' + dl_list[1]
data2 = gzip.open(file_path2,'rb')
data2= data2.read()
train_label = np.frombuffer(data2,np.uint8,offset=8)


file_path3 = dataset_dir + '/' + dl_list[2]
data3 = gzip.open(file_path3,'rb')
data3= data3.read()
data3 = np.frombuffer(data3,np.uint8,offset=16)
test_img = data3.reshape(-1,784)

file_path4 = dataset_dir + '/' + dl_list[3]
data4 = gzip.open(file_path4,'rb')
data4= data4.read()
test_label = np.frombuffer(data4,np.uint8,offset=8)

dataset=[train_img,train_label,test_img,test_label]