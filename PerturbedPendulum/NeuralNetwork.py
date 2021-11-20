import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import os
import sys
import random as rd

sys.path.append("..")

from AllMethods import PerturbedPendulum_Methods as pp

#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Import Data
x, y = pp.read_dataset('x_train.txt', 'y_train.txt')

#Training Data
train_dataset, val_dataset = pp.train_dataset(x, y, 20, 180, 20) #val_len, train_batch, val_batch


#Models
class MyModel1(tf.keras.Model):

    def __init__(self):
        super(MyModel1, self).__init__()
        self.dense1 = tf.keras.layers.Dense(2, activation=tf.nn.tanh)
        
        self.dense2 = tf.keras.layers.Dense(12, activation=tf.nn.tanh)
        self.dense3 = tf.keras.layers.Dense(12, activation=tf.nn.tanh)
        
        self.dense4 = tf.keras.layers.Dense(2, activation=tf.nn.tanh)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)
    
class MyModel2(tf.keras.Model):

    def __init__(self):
        super(MyModel2, self).__init__()
        self.dense1 = tf.keras.layers.Dense(2, activation=tf.nn.tanh)
        
        self.dense2 = tf.keras.layers.Dense(20, activation=tf.nn.tanh)
        self.dense3 = tf.keras.layers.Dense(20, activation=tf.nn.tanh)
        self.dense4 = tf.keras.layers.Dense(20, activation=tf.nn.tanh)
        
        self.dense5 = tf.keras.layers.Dense(2, activation=tf.nn.tanh)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)

        return self.dense5(x)
    

class MyModel3(tf.keras.Model):

    def __init__(self):
        super(MyModel3, self).__init__()
        self.dense1 = tf.keras.layers.Dense(2, activation=tf.nn.tanh)
        
        self.dense2 = tf.keras.layers.Dense(30, activation=tf.nn.tanh)
        self.dense3 = tf.keras.layers.Dense(30, activation=tf.nn.tanh)
        self.dense4 = tf.keras.layers.Dense(30, activation=tf.nn.tanh)
        self.dense5 = tf.keras.layers.Dense(30, activation=tf.nn.tanh)

        self.dense6 = tf.keras.layers.Dense(2, activation=tf.nn.tanh)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)

        return self.dense6(x)
    

model1 = MyModel1()
model2 = MyModel2()
model3 = MyModel3()



#Decaying learning rate
# DECAYING LEARNING RATE
def scheduler(epoch, lr):
    if epoch%100==0:
        print("epoch: ", epoch)
    if epoch < 5:
        return lr
    else:
        return lr*tf.math.exp(-0.0002)
    
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


#CHECKPOINT 1,2,3
#checkpoint_path_1 = "checkpoints_NN_PP_1/cp.ckpt"
#checkpoint_dir_1 = os.path.dirname(checkpoint_path_1)

#checkpoint_path_2 = "checkpoints_NN_PP_2/cp.ckpt"
#checkpoint_dir_2 = os.path.dirname(checkpoint_path_2)

checkpoint_path_3 = "checkpoints_NN_PP_3/cp.ckpt"
checkpoint_dir_3 = os.path.dirname(checkpoint_path_3)


#Callbacks
#cp_callback_1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_1,
#                                                 save_weights_only=True,
#                                                 save_best_only=True,
#                                                 verbose=0)
#Callbacks
#cp_callback_2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_2,
#                                                 save_weights_only=True,
#                                                 save_best_only=True,
#                                                 verbose=0)
#Callbacks
cp_callback_3 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_3,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=0)

#model1.compile(
#    loss= tf.keras.losses.MeanSquaredError(),
#    optimizer=keras.optimizers.Adam(0.01), 
#    metrics=["accuracy"],
#)

#model2.compile(
#    loss= tf.keras.losses.MeanSquaredError(),
#    optimizer=keras.optimizers.Adam(0.01), 
#    metrics=["accuracy"],
#)

model3.compile(
    loss= tf.keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(0.01),
    metrics=["accuracy"],
)

Epochs = 3000

#history1 = model1.fit(train_dataset, epochs = Epochs, validation_data=val_dataset, callbacks=[callback, cp_callback_1], verbose=0)
#history2 = model2.fit(train_dataset, epochs = Epochs, validation_data=val_dataset, callbacks=[callback, cp_callback_2], verbose=0)
history3 = model3.fit(train_dataset, epochs = Epochs, validation_data=val_dataset, callbacks=[callback, cp_callback_3], verbose=0)



############## PLOT LOSS ##################

plt.figure(figsize = (13,6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'Unstructured Neural Networks', fontsize = 30)

#plt.plot(history1.history["loss"], label="Training loss $\mathbf{H}_{4}^{10}$")
#plt.plot(history1.history["val_loss"], label="Validation loss $\mathbf{H}_{4}^{10}$")

#plt.plot(history2.history["loss"], label="Training loss $\mathbf{H}_{5}^{20}$")
#plt.plot(history2.history["val_loss"], label="Validation loss $\mathbf{H}_{5}^{20}$")

plt.plot(history3.history["loss"], label="Training loss $\mathbf{H}_{NN}$")
plt.plot(history3.history["val_loss"], label="Validation loss $\mathbf{H}_{NN}$")

plt.xlabel(r'Epoch', fontsize=20)
plt.ylabel(r'Loss', fontsize=20)
plt.yscale("log")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=17)
plt.grid(axis='both', alpha=.3)
plt.savefig("Loss_NN_PP")


################ PLOT ITERATION ##############
N = 300
x0 = np.array([[rd.randrange(-400, 400, 1)*0.001, rd.randrange(-400, 400, 1)*0.001] for _ in range(N)])
iterations = [model3(x0)]
for _ in range(200):
    iterations.append(model3(iterations[-1]))
    
plt.figure(figsize=(10,10))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'Unstructured Neural Networks', fontsize = 25)

colors = []
sizes = []
for i in range(N):
    if i==18:
        colors.append("r")
        sizes.append(10)
    else:
        colors.append("b")
        sizes.append(1)

plt.scatter(*zip(*iterations[0].numpy()), s=1, linewidth=0, color="b", label = "Iterations")

        
for i in iterations:
    plt.scatter(*zip(*i.numpy()), s=1, linewidth=0, color="b")
    
plt.scatter(*zip(*x0), s=10, linewidth=0, color="r", label = "Starting points")


plt.xlabel(r'$q$', fontsize=28, labelpad=8)
plt.ylabel(r'$p$', fontsize=28, labelpad=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(axis='both', alpha=.3)
lgnd = plt.legend(loc="upper right", numpoints=1, fontsize=20)
lgnd.legendHandles[0]._sizes = [15]
lgnd.legendHandles[1]._sizes = [30]
plt.savefig("Test_NN_PP")
plt.show()


################ PLOT REVERSIBILITY ##########################

x0 = np.array([[rd.randrange(250, 300, 1)*0.001, rd.randrange(250, 300, 1)*0.001] for _ in range(100)])

f_iterations = [model3(x0)]
for _ in range(100):
    f_iterations.append(model3(f_iterations[-1]))

b_iterations = [f_iterations[-1]*np.array([[1., -1.]])]

for _ in range(100):
    b_iterations.append(model3(b_iterations[-1]))

plt.figure(figsize=(10,10))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'Unstructured Neural Network', fontsize = 25)


plt.scatter(*zip(*f_iterations[0].numpy()), s=1, linewidth=0, color='r', label= r'$\hat{\mathcal{T}}^{n}[X_{0}] \qquad n=1,\dots ,100$')

for i in f_iterations:
    plt.scatter(*zip(*i.numpy()), s=1, linewidth=0, color='r')

b_iterations_symm = b_iterations*np.array([[1., -1.]])

plt.scatter(*zip(*b_iterations_symm[0]), s=1, linewidth=0, color='b', label= r'$R \hat{\mathcal{T}}^{n}[R(\hat{\mathcal{T}}^{100}(X_{0}))] \qquad n=1,\dots 100$')

for i in b_iterations_symm:
    plt.scatter(*zip(*i), s=1, linewidth=0, color='b')

plt.xlabel(r'$q$', fontsize=28, labelpad=8)
plt.ylabel(r'$p$', fontsize=28, labelpad=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(axis='both', alpha=.3)
lgnd = plt.legend(scatterpoints=1, fontsize=25)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.savefig("Test_Rev_NN_PP")
plt.show()


#Quantitative Test
A = np.array(f_iterations)
B = np.flip(b_iterations_symm, 0)

test = (np.square(A - B)).mean(axis=None)

f = open("QRev_NN_PP.txt", "a")
f.write(str(test))
f.close()


################# Prediction.txt ##############################

N_predictions = 100

x0 = np.array([[0.0, 0.1],[0.0, 0.2],[0.0, 0.3],[0.0, 0.4],[0.0, 0.5],[0.0, 0.6],[0.1, 0.],[0.2, 0.],[0.3, 0.],[0.4, 0.],[0.5, 0.]])


#iterations1 = [model1(x0)]
#for _ in range(N_predictions):
#    iterations1.append(model1(iterations1[-1]))

#iterations2 = [model2(x0)]
#for _ in range(N_predictions):
#    iterations2.append(model2(iterations2[-1]))

iterations3 = [model3(x0)]
for _ in range(N_predictions):
    iterations3.append(model3(iterations3[-1]))


# f = open("NN_PP_Prediction1.txt", "a")
# for i in iterations1:
#     for j in i.numpy():
#         f.write(str(j[0]))
#         f.write("\n")
#         f.write(str(j[1]))
#         f.write("\n")
# f.close()

# f = open("NN_PP_Prediction2.txt", "a")
# for i in iterations2:
#     for j in i.numpy():
#         f.write(str(j[0]))
#         f.write("\n")
#         f.write(str(j[1]))
#         f.write("\n")
# f.close()

f = open("NN_PP_Prediction3.txt", "a")
for i in iterations3:
    for j in i.numpy():
        f.write(str(j[0]))
        f.write("\n")
        f.write(str(j[1]))
        f.write("\n")
f.close()
