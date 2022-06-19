import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import pickle
from struct import unpack
from brian2 import *

 
def data_loading(file_name, is_train = True):
    if os.path.isfile('%s.pickle' % file_name):
        data = pickle.load(open('%s.pickle' % file_name,'rb'))
    else:
        if is_train:
            images = open('./Data/train-images.idx3-ubyte','rb')
            labels = open('./Data/train-labels.idx1-ubyte','rb')
        else:
            images = open('./Data/t10k-images.idx3-ubyte','rb')
            labels = open('./Data/t10k-labels.idx1-ubyte','rb')
        images.read(4)  
        image_num = unpack('>I', images.read(4))[0]
        r = unpack('>I', images.read(4))[0]
        c = unpack('>I', images.read(4))[0]
        labels.read(4)  
        N = unpack('>I', labels.read(4))[0]
        x = np.zeros((N, r, c), dtype=np.uint8)  
        y = np.zeros((N, 1), dtype=np.uint8)  
        for i in range(N):
            x[i] = [[unpack('>B', images.read(1))[0] for uc in range(c)]  for ur in range(r) ]
            y[i] = unpack('>B', labels.read(1))[0]
        data = {'x': x, 'y': y, 'rows': r, 'cols': c}
        pickle.dump(data, open("%s.pickle" % file_name, "wb"))
    return data
def rank_getting(assi, sp_rate):
    rate_sum = [0] * 10
    assi_num = [0] * 10
    for i in range(10):
        assi_num[i] = len(np.where(assi == i)[0])
        if assi_num[i] > 0:
            rate_sum[i] = np.sum(sp_rate[assi == i]) / assi_num[i]
    return np.argsort(rate_sum)[::-1]

def assig_getting(results, in_num):
    assi = np.zeros(n_e)
    in_nums = np.asarray(in_num)
    rate_maxx = [0] * n_e
    for j in range(10):
        assi_num = len(np.where(in_nums == j)[0])
        if assi_num > 0:
            rate = np.sum(results[in_nums == j], axis = 0) / assi_num
        for i in range(n_e):
            if rate[i] > rate_maxx[i]:
                rate_maxx[i] = rate[i]
                assi[i] = j
    return assi

MNIST_path = './'
data_path = './activity/'
train_suff = '1000'
test_suff = '1000'
train_st_t = 0
train_ed_t = int(train_suff)
test_st_t = 0
test_ed_t = int(test_suff)

n_e = 400

training = data_loading(MNIST_path + 'training')
testing = data_loading(MNIST_path + 'testing', is_train = False)

prefix = "v3_10_"
train_re_moni = np.load(data_path + prefix + 'resultPopVecs' + train_suff  + '.npy')
train_in_moni = np.load(data_path + prefix + 'inputNumbers' + train_suff + '.npy')
test_re_moni = np.load(data_path + prefix + 'resultPopVecs' + test_suff + '.npy')
test_in_moni = np.load(data_path + prefix + 'inputNumbers' + test_suff + '.npy')

print('get assignment')
test_re = np.zeros((10, test_ed_t-test_st_t))
assi = assig_getting(train_re_moni[train_st_t:train_ed_t], 
                                  train_in_moni[train_st_t:train_ed_t])
num = 0 
num_tests = test_ed_t / 1000
accu_sum = [0] * int(num_tests)
while (num < num_tests):
    end_time = min(test_ed_t, 1000*(num+1))
    start_time = 10000*num
    test_re = np.zeros((10, end_time-start_time))
    for i in range(end_time - start_time):
        test_re[:,i] = rank_getting(assi, 
                                                          test_re_moni[i+start_time,:])
    difference = test_re[0,:] - test_in_moni[start_time:end_time]
    corr = len(np.where(difference == 0)[0])
    incorr = np.where(difference != 0)[0]
    accu_sum[num] = corr/float(end_time-start_time) * 100
    print('Sum response : ', accu_sum[num], ' num incorrect: ', len(incorr))
    num += 1
print('Sum response  mean: ', np.mean(accu_sum),  ' standard deviation: ', np.std(accu_sum))


show()
