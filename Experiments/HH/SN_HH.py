import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random as rd

sys.path.append("..")

from AllMethods import ReversibleSymplecticNN as rs
from AllMethods import HenonHeiles_Methods as hh

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

Ename = "008333"

x, y = hh.read_dataset('x_train_'+ Ename +'.txt', 'y_train_'+ Ename +'.txt')

train_dataset, val_dataset = hh.train_dataset(x, y, 10000, 50000, 10000)


model1R = rs.SympNet(15, 8, 'reversible')
model2R = rs.SympNet(40, 8, 'reversible')

model1N = rs.SympNet(15, 8, 'non_reversible')
model2N = rs.SympNet(40, 8, 'non_reversible')

model1N2 = rs.SympNet(30, 8, 'non_reversible')
#

# DECAYING LEARNING RATE
def scheduler(epoch, lr):
    if epoch%100==0:
        print("epoch: ", epoch)
    if epoch < 5:
        return lr
    else:
        return lr*tf.math.exp(-0.0007)
    
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


#CHECKPOINT 1,2,3
checkpoint_path_1R = "checkpoints_SN_PP_1R/cp.ckpt"
checkpoint_dir_1R = os.path.dirname(checkpoint_path_1R)

checkpoint_path_2R = "checkpoints_SN_PP_2R/cp.ckpt"
checkpoint_dir_2R = os.path.dirname(checkpoint_path_2R)

checkpoint_path_1N = "checkpoints_SN_PP_1N/cp.ckpt"
checkpoint_dir_1N = os.path.dirname(checkpoint_path_1N)

checkpoint_path_2N = "checkpoints_SN_PP_2N/cp.ckpt"
checkpoint_dir_2N = os.path.dirname(checkpoint_path_2N)

checkpoint_path_1N2 = "checkpoints_SN_PP_1N2/cp.ckpt"
checkpoint_dir_1N2 = os.path.dirname(checkpoint_path_1N2)



#Callbacks
cp_callback_1R = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_1R,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=0)
#Callbacks
cp_callback_2R = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_2R,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=0)
#Callbacks
cp_callback_1N = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_1N,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=0)
#Callbacks
cp_callback_2N = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_2N,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=0)
#Callbacks
cp_callback_1N2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_1N2,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=0)



model1R.compile(
    loss= tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=["accuracy"],
)

model2R.compile(
    loss= tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=["accuracy"],
)

model1N.compile(
    loss= tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=["accuracy"],
)

model2N.compile(
    loss= tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=["accuracy"],
)

model1N2.compile(
    loss= tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=["accuracy"],
)



Epochs = 3000

history1R = model1R.fit(train_dataset, epochs = Epochs, validation_data=val_dataset, callbacks=[callback, cp_callback_1R], verbose=0)
history2R = model2R.fit(train_dataset, epochs = Epochs, validation_data=val_dataset, callbacks=[callback, cp_callback_2R], verbose=0)
history1N = model1N.fit(train_dataset, epochs = Epochs, validation_data=val_dataset, callbacks=[callback, cp_callback_1N], verbose=0)
history2N = model2N.fit(train_dataset, epochs = Epochs, validation_data=val_dataset, callbacks=[callback, cp_callback_2N], verbose=0)
history1N2 = model1N2.fit(train_dataset, epochs = Epochs, validation_data=val_dataset, callbacks=[callback, cp_callback_1N2], verbose=0)




############## PLOT LOSS 1 ##################

plt.figure(figsize = (13,6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'SympNets', fontsize = 30)

plt.plot(history1R.history["loss"], label="Training loss $\mathbf{H}_{R, 15}^{\text{SympNet}}$")
plt.plot(history1R.history["val_loss"], label="Validation loss $\mathbf{H}_{R, 15}^{\text{SympNet}}$")

plt.plot(history1N.history["loss"], label="Training loss $\mathbf{H}_{15}^{\text{SympNet}}$")
plt.plot(history1N.history["val_loss"], label="Validation loss $\mathbf{H}_{15}^{\text{SympNet}}$")

plt.plot(history1N2.history["loss"], label="Training loss $\mathbf{H}_{30}^{\text{SympNet}}$")
plt.plot(history1N2.history["val_loss"], label="Validation loss $\mathbf{H}_{30}^{\text{SympNet}}$")

plt.xlabel(r'Epoch', fontsize=20)
plt.ylabel(r'Loss', fontsize=20)
plt.yscale("log")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=17)
plt.grid(axis='both', alpha=.3)
plt.savefig("Loss_SN_PP_1Rvs1Nvs1N2")

############## PLOT LOSS 2 ##################

plt.figure(figsize = (13,6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'SympNets', fontsize = 30)

plt.plot(history2R.history["loss"], label="Training loss $\mathbf{H}_{R, 40}^{\text{SympNet}}$")
plt.plot(history2R.history["val_loss"], label="Validation loss $\mathbf{H}_{R, 40}^{\text{SympNet}}$")

plt.plot(history2N.history["loss"], label="Training loss $\mathbf{H}_{40}^{\text{SympNet}}$")
plt.plot(history2N.history["val_loss"], label="Validation loss $\mathbf{H}_{40}^{\text{SympNet}}$")


plt.xlabel(r'Epoch', fontsize=20)
plt.ylabel(r'Loss', fontsize=20)
plt.yscale("log")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=17)
plt.grid(axis='both', alpha=.3)
plt.savefig("Loss_SN_PP_2Rvs2N")



################ PLOT ITERATION 1 ##############
N = 300
x0 = np.array([[rd.randrange(100, 400, 1)*0.001, rd.randrange(-100, 100, 1)*0.001] for _ in range(N)])
iterations = [model2R(x0)]
for _ in range(200):
    iterations.append(model2R(iterations[-1]))
    
plt.figure(figsize=(10,10))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'SympNet', fontsize = 25)

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
plt.savefig("Test_SN_PP_2R")
plt.show()

################ PLOT ITERATION 2 ##############
N = 300
x0 = np.array([[rd.randrange(100, 400, 1)*0.001, rd.randrange(-100, 100, 1)*0.001] for _ in range(N)])
iterations = [model2N(x0)]
for _ in range(200):
    iterations.append(model2N(iterations[-1]))
    
plt.figure(figsize=(10,10))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'SympNet', fontsize = 25)

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
plt.savefig("Test_SN_PP_2N")
plt.show()





################ PLOT REVERSIBILITY 1 ##########################

x0 = np.array([[rd.randrange(300, 400, 1)*0.001, rd.randrange(50, 100, 1)*0.001] for _ in range(100)])

f_iterations = [model2R(x0)]
for _ in range(100):
    f_iterations.append(model2R(f_iterations[-1]))

b_iterations = [f_iterations[-1]*np.array([[1., -1.]])]

for _ in range(100):
    b_iterations.append(model2R(b_iterations[-1]))

plt.figure(figsize=(10,10))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'Time-Reversible SympNet', fontsize = 25)


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
plt.savefig("Test_Rev_SN_PP_2R")
plt.show()


#Quantitative Test
A = np.array(f_iterations)
B = np.flip(b_iterations_symm, 0)

test = (np.square(A - B)).mean(axis=None)

f = open("QRev_SN_PP_2R.txt", "a")
f.write(str(test))
f.close()


################ PLOT REVERSIBILITY 2 ##########################

x0 = np.array([[rd.randrange(300, 400, 1)*0.001, rd.randrange(50, 100, 1)*0.001] for _ in range(100)])

f_iterations = [model2N(x0)]
for _ in range(100):
    f_iterations.append(model2N(f_iterations[-1]))

b_iterations = [f_iterations[-1]*np.array([[1., -1.]])]

for _ in range(100):
    b_iterations.append(model2N(b_iterations[-1]))

plt.figure(figsize=(10,10))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'SympNet', fontsize = 25)


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
plt.savefig("Test_Rev_SN_PP_2N")
plt.show()


#Quantitative Test
A = np.array(f_iterations)
B = np.flip(b_iterations_symm, 0)

test = (np.square(A - B)).mean(axis=None)

f = open("QRev_SN_PP_2N.txt", "a")
f.write(str(test))
f.close()


################# Prediction.txt ##############################

N_predictions = 100

x0 = np.array([[0.1, 0.0],[0.2, 0.0],[0.4, 0.0],[0.0, 0.05], [0.0, 0.1]])


iterations2R = [model2R(x0)]
for _ in range(N_predictions):
    iterations2R.append(model2R(iterations2R[-1]))

iterations2N = [model2N(x0)]
for _ in range(N_predictions):
    iterations2N.append(model2N(iterations2N[-1]))



f = open("SN_PP_Prediction2R.txt", "a")
for i in iterations2R:
    for j in i.numpy():
        f.write(str(j[0]))
        f.write("\n")
        f.write(str(j[1]))
        f.write("\n")
f.close()

f = open("SN_PP_Prediction2N.txt", "a")
for i in iterations2N:
    for j in i.numpy():
        f.write(str(j[0]))
        f.write("\n")
        f.write(str(j[1]))
        f.write("\n")
f.close()
