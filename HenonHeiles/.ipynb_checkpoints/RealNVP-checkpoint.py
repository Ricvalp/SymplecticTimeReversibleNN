import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import random as rd
import sys
import os

sys.path.append("..")

from AllMethods import HenonHeiles_Methods as hh

#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


#E = 0.1
Ename = "008333"

#Import Data
x, y = hh.read_dataset('x_train_'+ Ename +'.txt', 'y_train_'+ Ename +'.txt')

#Training Data
train_dataset, val_dataset = hh.train_dataset(x, y, 50, 800, 50) #val_len, train_batch, val_batch


#MLPs #####################

class MLP1(tf.keras.layers.Layer):
    
    def __init__(self):
        super(MLP1, self).__init__()

    def build(self, input_shape):
        
        self.L1 = tf.keras.layers.Dense(3, activation='tanh')
        self.L2 = tf.keras.layers.Dense(3, activation='tanh')
        self.L3 = tf.keras.layers.Dense(2, activation='tanh')


    def call(self, inputs, training=False):
        
        h = self.L1(inputs)
        h = self.L2(h)
        h = self.L3(h)
        
        return h



class MLP2(tf.keras.layers.Layer):
    
    def __init__(self):
        super(MLP2, self).__init__()

    def build(self, input_shape):
        
        self.L1 = tf.keras.layers.Dense(6, activation='tanh')
        self.L2 = tf.keras.layers.Dense(6, activation='tanh')
        self.L3 = tf.keras.layers.Dense(6, activation='tanh')
        self.L4 = tf.keras.layers.Dense(2, activation='tanh')


    def call(self, inputs, training=False):
        
        h = self.L1(inputs)
        h = self.L2(h)
        h = self.L3(h)
        h = self.L4(h)
        
        return h



class MLP3(tf.keras.layers.Layer):
    
    def __init__(self):
        super(MLP3, self).__init__()

    def build(self, input_shape):
        
        self.L1 = tf.keras.layers.Dense(12, activation='tanh')
        self.L2 = tf.keras.layers.Dense(12, activation='tanh')
        self.L3 = tf.keras.layers.Dense(12, activation='tanh')
        self.L4 = tf.keras.layers.Dense(12, activation='tanh')
        self.L5 = tf.keras.layers.Dense(2, activation='tanh')


    def call(self, inputs, training=False):
        
        h = self.L1(inputs)
        h = self.L2(h)
        h = self.L3(h)
        h = self.L4(h)
        h = self.L5(h)

        return h


####################################################################
####################################################################
####################################################################

def get_shift_and_log_scale_resnet(input_shape, blocks, shift):
    
    inputs = tf.keras.Input(shape=input_shape)
    h = inputs

    for block in blocks:
        h = block(h)

    shift, log_scale = shift(inputs), h
    log_scale = tf.math.tanh(log_scale)
    return Model(inputs=inputs, outputs=[shift, log_scale], name='name')
    
    
    
class AffineCouplingLayer(layers.Layer):

    def __init__(self, shift_and_log_scale_fn, mask):
        
        super(AffineCouplingLayer, self).__init__()
        self.shift_and_log_scale_fn = shift_and_log_scale_fn
        self.b = tf.cast(mask, tf.float32)

    def call(self, x, inverse):
        
        if inverse == 1:
            t, log_s = self.shift_and_log_scale_fn(x * self.b)
            y = self.b * x + (1 - self.b) * (x * tf.exp(log_s) + t)
            return y
        
        if inverse == 0:
            t, log_s = self.shift_and_log_scale_fn(x * self.b)
            y = self.b * x + (1 - self.b) * ((x - t) * tf.exp(-log_s))
            return y


class R(tf.keras.layers.Layer):

    def __init__(self):
        super(R, self).__init__()
        
    def call(self, inputs):
        return inputs*tf.constant([[1., -1.]])
    

class RealNVPModel(tf.keras.Model):
    
    def __init__(self, shift_and_log_scale, **kwargs):
        
        super(RealNVPModel, self).__init__()
        self.R = R()
        masks = []
        self.l = []
        
        for j in range(len(shift_and_log_scale)):
            
            if j%2==0:
                masks.append(tf.constant([[0., 1.]]))
            else:
                masks.append(tf.constant([[1., 0.]]))

        
        for i,j in zip(shift_and_log_scale, masks):
            self.l.append(AffineCouplingLayer(i, j))
        

    def call(self, input_tensor):
        
        a = self.R(input_tensor)
        
        for layer in self.l:
            a = layer(a, 1)

        a = self.R(a)

        for layer in self.l[::-1]:
            a = layer(a, 0)
        
        return a
    
###############################################################################
    
    
num_blocks = 6

blocks1 = [MLP1() for i in range(num_blocks)]
shifts1 = [MLP1() for i in  range(num_blocks)]
shift_and_log_scale1 = [get_shift_and_log_scale_resnet((2), [i], j) for i,j in zip(blocks1, shifts1)]
model1 = RealNVPModel(shift_and_log_scale1)


blocks2 = [MLP2() for i in range(num_blocks)]
shifts2 = [MLP2() for i in  range(num_blocks)]
shift_and_log_scale2 = [get_shift_and_log_scale_resnet((2), [i], j) for i,j in zip(blocks2, shifts2)]
model2 = RealNVPModel(shift_and_log_scale2)

num_blocks = 6

blocks3 = [MLP3() for i in range(num_blocks)]
shifts3 = [MLP3() for i in  range(num_blocks)]
shift_and_log_scale3 = [get_shift_and_log_scale_resnet((2), [i], j) for i,j in zip(blocks3, shifts3)]
model3 = RealNVPModel(shift_and_log_scale3)


###############################################################################

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
#checkpoint_path_1 = "checkpoints_RNN_HH_1/cp.ckpt"
#checkpoint_dir_1 = os.path.dirname(checkpoint_path_1)

checkpoint_path_2 = "checkpoints_RNN_HH_2/cp.ckpt"
checkpoint_dir_2 = os.path.dirname(checkpoint_path_2)

#checkpoint_path_3 = "checkpoints_RNN_HH_3/cp.ckpt"
#checkpoint_dir_3 = os.path.dirname(checkpoint_path_3)




#Callbacks
#cp_callback_1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_1,
#                                                 save_weights_only=True,
#                                                 save_best_only=True,
#                                                 verbose=0)
#Callbacks
cp_callback_2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_2,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=0)
#Callbacks
#cp_callback_3 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_3,
#                                                 save_weights_only=True,
#                                                 save_best_only=True,
#                                                 verbose=0)



#model1.compile(
#    loss= tf.keras.losses.MeanSquaredError(),
#    optimizer=tf.keras.optimizers.Adam(0.01),
#    metrics=["accuracy"],
#)

model2.compile(
    loss= tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=["accuracy"],
)

#model3.compile(
#    loss= tf.keras.losses.MeanSquaredError(),
#    optimizer=tf.keras.optimizers.Adam(0.01),
#    metrics=["accuracy"],
#)



Epochs = 10000

#history1 = model1.fit(train_dataset, epochs = Epochs, validation_data=val_dataset, callbacks=[callback, cp_callback_1], verbose=0)
history2 = model2.fit(train_dataset, epochs = Epochs, validation_data=val_dataset, callbacks=[callback, cp_callback_2], verbose=0)
#history3 = model3.fit(train_dataset, epochs = Epochs, validation_data=val_dataset, callbacks=[callback, cp_callback_3], verbose=0)




############## PLOT LOSS ##################

plt.figure(figsize = (13,6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'Time-Reversible Compositions of Bijective Maps', fontsize = 30)

#plt.plot(history1.history["loss"], label="Training loss $\mathbf{H}_{R}^{1}$")
#plt.plot(history1.history["val_loss"], label="Validation loss $\mathbf{H}_{R}^{1}$")

plt.plot(history2.history["loss"], label="Training loss $\mathbf{H}'_{R}$")
plt.plot(history2.history["val_loss"], label="Validation loss $\mathbf{H}'_{R}$")

#plt.plot(history3.history["loss"], label="Training loss $\mathbf{H}_{R}^{3}$")
#plt.plot(history3.history["val_loss"], label="Validation loss $\mathbf{H}_{R}^{3}$")

plt.xlabel(r'Epoch', fontsize=20)
plt.ylabel(r'Loss', fontsize=20)
plt.yscale("log")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=17)
plt.grid(axis='both', alpha=.3)
plt.savefig("Loss_RNN_HH")


################ PLOT ITERATION ##############

N = 300
x0 = np.array([[rd.randrange(100, 400, 1)*0.001, rd.randrange(-100, 100, 1)*0.001] for _ in range(N)])
iterations = [model2(x0)]
for _ in range(200):
    iterations.append(model2(iterations[-1]))
    
plt.figure(figsize=(10,10))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'Time-Reversible Composition of Bijective Maps', fontsize = 25)

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
plt.savefig("Test_RNN_HH")
plt.show()


################ PLOT REVERSIBILITY ##########################

x0 = np.array([[rd.randrange(300, 400, 1)*0.001, rd.randrange(50, 100, 1)*0.001] for _ in range(100)])

f_iterations = [model2(x0)]
for _ in range(100):
    f_iterations.append(model2(f_iterations[-1]))

b_iterations = [f_iterations[-1]*np.array([[1., -1.]])]

for _ in range(100):
    b_iterations.append(model2(b_iterations[-1]))

plt.figure(figsize=(10,10))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'Time-Reversible Composition of Bijective Maps', fontsize = 25)


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
plt.savefig("Test_Rev_RNN_HH")
plt.show()


#Quantitative Test
A = np.array(f_iterations)
B = np.flip(b_iterations_symm, 0)

test = (np.square(A - B)).mean(axis=None)

f = open("QRev_RNN_HH.txt", "a")
f.write(str(test))
f.close()


################# Prediction.txt ##############################

N_predictions = 100

x0 = np.array([[0.1, 0.0],[0.2, 0.0],[0.4, 0.0],[0.0, 0.05], [0.0, 0.1]])


#iterations1 = [model1(x0)]
#for _ in range(N_predictions):
#    iterations1.append(model1(iterations1[-1]))

iterations2 = [model2(x0)]
for _ in range(N_predictions):
    iterations2.append(model2(iterations2[-1]))

#iterations3 = [model3(x0)]
#for _ in range(N_predictions):
#iterations3.append(model3(iterations1[-1]))


f = open("RNN_HH_Prediction1.txt", "a")
for i in iterations1:
    for j in i.numpy():
        f.write(str(j[0]))
        f.write("\n")
        f.write(str(j[1]))
        f.write("\n")
f.close()

f = open("RNN_HH_Prediction2.txt", "a")
for i in iterations2:
    for j in i.numpy():
        f.write(str(j[0]))
        f.write("\n")
        f.write(str(j[1]))
        f.write("\n")
f.close()

f = open("RNN_HH_Prediction3.txt", "a")
for i in iterations3:
    for j in i.numpy():
        f.write(str(j[0]))
        f.write("\n")
        f.write(str(j[1]))
        f.write("\n")
f.close()
