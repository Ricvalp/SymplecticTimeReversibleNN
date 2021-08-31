import tensorflow as tf
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


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
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

model = MyModel()


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

checkpoint_path_NN = "checkpoints_NN/cp.ckpt"
checkpoint_dir_NN = os.path.dirname(checkpoint_path_NN)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_NN,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=0)


################################################################################

model.compile(
    loss= tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.002),
    metrics=["accuracy"],
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs = 10000, callbacks = [callback, cp_callback], verbose=1)

################################################################################

f = open("NN_loss.txt", "a")
for i in history.history["loss"]:
    f.write(str(i))
    f.write("\n")
f.close()

f = open("NN_val.txt", "a")
for i in history.history["NN_loss"]:
    f.write(str(i))
    f.write("\n")
f.close()

################################################################################

x0 = x_train[100]
f_iterations = [model(x0)]
for _ in range(500):
    f_iterations.append(model(f_iterations[-1]))

f = open("3D_NN_Prediction.txt", "a")
for i in f_iterations:
    f.write(str(i[0][0].numpy()))
    f.write("\n")
    f.write(str(i[0][1].numpy()))
    f.write("\n")
    f.write(str(i[0][2].numpy()))
    f.write("\n")
f.close()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    