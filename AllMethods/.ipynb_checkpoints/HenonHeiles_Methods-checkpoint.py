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
    
    x = []
    y = []
    
    read_x = open(file_name1, "r")
    read_y = open(file_name2, "r")
    
    
    for i in range(int(line_count/2)):
        
        x.append([float(read_x.readline()), float(read_x.readline())])
        
        y.append([float(read_y.readline()), float(read_y.readline())])
    
    read_x.close()
    read_y.close()
    
    print("Number of points: ", len(x))
    
    return x, y



def plot_dataset(x):
    
    plt.figure(figsize=(10,10))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(r'H\'enon Heiles, x = 0', fontsize = 25)
    plt.scatter(*zip(*x), color = "b", label = "target", linewidth = 0, s=2)
    plt.xlabel(r'$y$', fontsize=28, labelpad=8)
    plt.ylabel(r'$p_{y}$', fontsize=28, labelpad=15)
    plt.grid(axis='both', alpha=.3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)    
    #plt.legend(fontsize = 25, loc = 'upper right')
    plt.show()
    


def train_dataset(x, y, val_len, train_batch, val_batch):
    
    # TRAINING SET
    
    x_train = np.array(x)
    x_train = x_train.astype(np.float64)
    
    y_train = np.array(y)
    y_train = y_train.astype(np.float64)
    
    
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
