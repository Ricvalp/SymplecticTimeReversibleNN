import tensorflow as tf
from tensorflow import keras
import time
import os
import sys

sys.path.append("..")

from AllMethods import ReversibleSymplecticNN as rs
from AllMethods import HenonHeiles_Methods as hh


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


x, y = hh.read_dataset('x_train.txt', 'y_train.txt')

train_dataset, val_dataset = hh.train_dataset(x, y, 1000, 1000, 1000) # val_len, train_batch, val_batch

model = rs.Henon(50, 'reversible')

# DECAYING LEARNING RATE
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr*tf.math.exp(-0.001)
    
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)



checkpoint_path = "checkpoints_Reversible_Henon/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,                                                 
                                                 verbose=1)


model.compile(
    loss= tf.keras.losses.MeanSquaredError(),
    #optimizer=tfps.optimizers.bfgs_minimize(), #clipvalue = 0.001),
    #optimizer=keras.optimizers.SGD(0.00001), #, clipvalue = 0.001),
    optimizer=keras.optimizers.Adam(0.003), #, clipvalue = 0.001),
    metrics=["accuracy"],
)


#wandb.init()
start_time = time.time()
history = model.fit(train_dataset, epochs = 500, validation_data=val_dataset, callbacks=[callback, cp_callback], verbose=1)
print("running time : %s seconds" % (time.time() - start_time))