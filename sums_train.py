import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow import keras


def load_data():
    '''
    Returns contents of data{0..1}.npy
    '''
    train_data = np.zeros((1, 40, 168))
    train_labels = np.zeros((1,))
    #for i in range(2):
    i = int(sys.argv[1])
    tmp_data = np.load('./Data/data{}.npy'.format(i))
    tmp_labels = np.load('./Data/lab{}.npy'.format(i))        
    train_data = np.concatenate((train_data, tmp_data), axis = 0)
    train_labels = np.concatenate((train_labels, tmp_labels), axis = 0)
    return train_data[1:], train_labels[1:]


# del train_data, train_labels
train_data, train_labels = load_data()
MNIST = tf.keras.models.load_model('./cnn.h5')

def predict(x):
    '''
    Arguments : 28 * 28 image
    Output : 10 * 1 vector
    '''
    m, i, j = x.shape
    x = x.reshape(m, i, j, 1).astype('float32') / 255
    out = MNIST.predict(x)
    return out

def Convolutional_MNIST(inp):
    '''
    Args: 
    inp => argument of size 40 * 168
    outp => numpy array of size 13 * 141 * 10
    '''
    k, m, n = inp.shape
    ki = 28
    kj = 28
#     img_no = 3

    stridei = 1
    stridej = 18
    
    outx = (m - ki) // (stridei) + 1
    outy = (n - kj) // (stridej) + 1
    
    rows = (m - ki) // (stridei) + 1
    cols = (n - kj) // (stridej) + 1
    
    outp = np.zeros((k, outx, outy, 10))
    
    for i in range(0, m - ki + 1, stridei):
        for j in range(0, n - kj + 1, stridej):
            
            img = inp[:, i:i + ki, j:j + kj]
            pres = predict(img)
            outi = i // stridei
            outj = j // stridej
            outp[:, outi, outj] = pres
    return outp


def batch_pred(data, start, end):
    k = train_data[start:end]
    h = Convolutional_MNIST(k)
    q = np.max(h, axis = 1)
    q.sort(axis = 1)
    p = q[:,-5:-1, :]
    l = p.argmax(axis = 2)
    final = l.sum(axis = 1)
    return final


final_sums = []

strides = 100
for i in range(0, 10000, strides):
    tmp_batch_sum = batch_pred(train_data, i, i + strides)
    final_sums.extend(tmp_batch_sum)

def plot_metrics(labels, Sums, num):
    sums = np.array(Sums)
    acc = (labels[:num] == sums)
    loss = np.abs(labels[:num] - sums) 
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    
    data = np.arange(0, num)
    ax[0].scatter(data, loss)
    
    ax[0].set_xlabel("Samples")
    ax[0].set_ylabel("Loss")

    ax[1].scatter(data, acc)
    ax[1].set_xlabel("Samples")
    ax[1].set_ylabel("Accuracy")

    accuracy = np.around((np.count_nonzero(acc) / acc.shape[0]) * 100, 2)
    fig.suptitle("Accuracy = {}".format(accuracy))
    fig.tight_layout()


plot_metrics(train_labels, final_sums, 10000)
plt.savefig("train_data{}.png".format(int(sys.argv[1])))
