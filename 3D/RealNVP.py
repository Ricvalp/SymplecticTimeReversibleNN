import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import numpy as np
import os

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


################################################################################

file = open("x_train.txt", "r")
line_count = 0
for line in file:
    if line != "\n":
        line_count += 1
file.close()

x = []
y = []

read_x = open("x_train.txt", "r")
read_y = open("y_train.txt", "r")
read_z = open("z_train.txt", "r")

for i in range(int(line_count)):

    x.append([float(read_x.readline()), float(read_y.readline()), float(read_z.readline())])

read_x.close()
read_y.close()
read_z.close()

print("Number of points: ", len(x))

################################################################################

x_train = np.array(x[0:-1])
x_train = x_train.astype(np.float32)

y_train = np.array(x[1:])
y_train = y_train.astype(np.float32)

val_len = 1000

x_val = x_train[-val_len:]
y_val = y_train[-val_len:]

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(1000)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(1000)


################################################################################
################################################################################
################################################################################


class MLP(tf.keras.layers.Layer):
    
    def __init__(self):
        super(MLP, self).__init__()


    def build(self, input_shape):
        
        
        self.L1 = tf.keras.layers.Dense(50, activation='tanh')
        self.L2 = tf.keras.layers.Dense(50, activation='tanh')
        self.L3 = tf.keras.layers.Dense(3, activation='tanh')
        
    def call(self, inputs, training=False):
        
        h = self.L1(inputs)
        h = self.L2(h)
        h = self.L3(h)
        
        return h


def get_shift_and_log_scale_resnet(input_shape, blocks, shift):
    
    inputs = tf.keras.Input(shape=input_shape)
    h = inputs

    for block in blocks:
        h = block(h)

    shift, log_scale = shift(inputs), h
    log_scale = tf.math.tanh(log_scale)
    return Model(inputs=inputs, outputs=[shift, log_scale], name='name')


class AffineCouplingLayer(tfb.Bijector):

    def __init__(self, shift_and_log_scale_fn, mask, **kwargs):
        super(AffineCouplingLayer, self).__init__(
            forward_min_event_ndims=1, **kwargs)
        self.shift_and_log_scale_fn = shift_and_log_scale_fn
        self.b = tf.cast(mask, tf.float32)

    def _forward(self, x):
        t, log_s = self.shift_and_log_scale_fn(x * self.b)
        y = self.b * x + (1 - self.b) * (x * tf.exp(log_s) + t)
        return y

    def _inverse(self, y):
        t, log_s = self.shift_and_log_scale_fn(y * self.b)
        x = self.b * y + (1 - self.b) * ((y - t) * tf.exp(-log_s))
        return x



class R(tf.keras.layers.Layer):

    def __init__(self):
        super(R, self).__init__()
        
    def call(self, inputs):
        return inputs*tf.constant([[-1., -1., -1.]])
    


class RealNVPModel(tf.keras.Model):
    
    def __init__(self, shift_and_log_scale, **kwargs):
        
        super(RealNVPModel, self).__init__()
        self.R = R()
        masks = []
        self.l = []
        
        for j in range(len(shift_and_log_scale)):
            
            if j%3==0:
                masks.append(tf.constant([[0., 1., 1.]]))
            if j%3==1:
                masks.append(tf.constant([[1., 0., 1.]]))
            else:
                masks.append(tf.constant([[1., 1., 0.]]))
        
        for i,j in zip(shift_and_log_scale, masks):
            self.l.append(AffineCouplingLayer(i, j))
        

    def call(self, input_tensor):
        
        a = self.R(input_tensor)
        
        for layer in self.l:
            a = layer._forward(a)

        a = self.R(a)

        for layer in self.l[::-1]:
            a = layer._inverse(a)
        
        return a


num_blocks = 5

blocks = [MLP() for i in range(num_blocks)]

shifts = [MLP() for i in range(len(blocks))]

shift_and_log_scale = [get_shift_and_log_scale_resnet((3), [i], j) for i,j in zip(blocks, shifts)]

model_rev = RealNVPModel(shift_and_log_scale)

################################################################################
################################################################################
################################################################################



# DECAYING LEARNING RATE
def scheduler(epoch, lr):
    if epoch < 5:
        print(lr)
        return lr
    else:
        #print(lr)
        return lr*tf.math.exp(-0.00009)
    
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

checkpoint_path_rev = "checkpoints_rev/cp.ckpt"
checkpoint_dir_rev = os.path.dirname(checkpoint_path_rev)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_rev,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=0)


################################################################################


model_rev.compile(
    loss= tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.002),
    metrics=["accuracy"],
)

history_rev = model_rev.fit(train_dataset, validation_data=val_dataset, epochs = 10000, callbacks = [callback, cp_callback], verbose=1)

################################################################################

f = open("Rev_loss.txt", "a")
for i in history_rev.history["loss"]:
    f.write(str(i))
    f.write("\n")
f.close()

f = open("Rev_val.txt", "a")
for i in history_rev.history["val_loss"]:
    f.write(str(i))
    f.write("\n")
f.close()

################################################################################

x0 = x_train[100]
f_iterations = [model_rev(x0)]
for _ in range(500):
    f_iterations.append(model_rev(f_iterations[-1]))

f = open("3D_Reversible_Prediction.txt", "a")
for i in f_iterations:
    f.write(str(i[0][0].numpy()))
    f.write("\n")
    f.write(str(i[0][1].numpy()))
    f.write("\n")
    f.write(str(i[0][2].numpy()))
    f.write("\n")
f.close()
