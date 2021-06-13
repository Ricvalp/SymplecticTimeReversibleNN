import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def read_dataset(file_name1, file_name2):
    
    file = open(file_name1, "r")
    line_count = 0
    for line in file:
        if line != "\n":
            line_count += 1
    file.close()
    
    
    Nlines = int(line_count/2)
    
    u0 = []
    
    prova = open(file_name1, "r")
    for i in range(Nlines):
        x = float(prova.readline())
        y = float(prova.readline())
        u0.append(np.array([x,y]))
    prova.close()
    
    T = []
    
    prova = open(file_name2, "r")
    
    for i in range(Nlines):
        x = float(prova.readline())
        y = float(prova.readline())
        T.append(np.array([x,y]))
    prova.close()
    
    return u0, T


def plot_dataset(u0, T):
    
    plt.figure(figsize=(10,10))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(r'Training Dataset', fontsize = 25)
    plt.scatter([u0[i][0] for i in range(len(u0))], [u0[i][1] for i in range(len(u0))], label = r'$x$', linewidth = 0, color ='r', s=1)
    plt.scatter([T[i][0] for i in range(len(u0))], [T[i][1] for i in range(len(u0))], label = r'$\mathcal{T}(x)$', linewidth = 0, color = 'b', s=1)
    plt.xlabel(r'$x$', fontsize=28, labelpad=8)
    plt.ylabel(r'$y$', fontsize=28, labelpad=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(axis='both', alpha=.3)
    plt.legend(fontsize = 25)
    plt.show()
    
def train_dataset(u0, T, val_len, train_batch, val_batch):
    
    x_train = np.array(u0)
    x_train = x_train.astype(np.float32)
    
    y_train = np.array(T)
    y_train = y_train.astype(np.float32)
    
    
    x_val = x_train[-val_len:]
    y_val = y_train[-val_len:]
    
    x_train = x_train[:-val_len]
    y_train = y_train[:-val_len]
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(val_batch)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(train_batch)
        
    return train_dataset, val_dataset


def export_loss(loss, val, num, overwrite):
    
    if overwrite==True:
        os.remove("Loss"+str(num)+".txt")
        os.remove("Val"+str(num)+".txt")

    f = open("Loss"+str(num)+".txt", "a")
    for i in loss:
        f.write(str(i))
        f.write("\n")
    f.close()
    
    #os.remove("Val2.txt")
    f = open("Val"+str(num)+".txt", "a")
    for i in val:
        f.write(str(i))
        f.write("\n")
    f.close()
