import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class linear_module_up(layers.Layer):
    
    def __init__(self):
        
        super().__init__()
        self.w = self.add_weight(
        name='w',
        shape=(1,),
        initializer = 'random_normal',
        trainable = True
        )
    

    def call(self, x, inverse):
        
        if inverse == 0:
        
            L = tf.concat([[[1., self.w[0]]],[[0., 1.]]], 0)
            return tf.linalg.matvec(L, x)
        
        else:
            
            L = tf.concat([[[1., -self.w[0]]],[[0., 1.]]], 0)
            return tf.linalg.matvec(L, x)


class linear_module_low(layers.Layer):
    
    def __init__(self):
        
        super().__init__()
        self.w = self.add_weight(
        name='w',
        shape=(1,),
        initializer = 'random_normal',
        trainable = True
        )

    def call(self, x, inverse):

        if inverse == 0:

                L = tf.concat([[[1., 0.]],[[self.w[0], 1.]]], 0)
                return tf.linalg.matvec(L,x)
        else:

                L = tf.concat([[[1., 0.]],[[-self.w[0], 1.]]], 0)
                return tf.linalg.matvec(L,x)



class bias(layers.Layer):
    
    def __init__(self):

        super().__init__()
        self.b = self.add_weight(
        name='b',
        shape=(2,),
        initializer = 'random_normal',
        trainable = True
        )

    def call(self, x, inverse):

        if inverse == 0:

                return x + self.b
        else:

                return x - self.b


            
class activation_module_up(layers.Layer):
    
    def __init__(self):
        
        super().__init__()
        self.w = self.add_weight(
        name='w',
        shape=(1,),
        initializer = 'random_normal',
        trainable = True
        )


    def call(self, x, inverse):
        
        if inverse == 0:
            
            t = tf.concat([[[0., self.w[0]]], [[0.,0.]]], 0)
            return x + tf.linalg.matvec(t, tf.math.tanh(x))

        else:
            
            t = tf.concat([[[0., -self.w[0]]], [[0.,0.]]], 0)
            return x + tf.linalg.matvec(t, tf.math.tanh(x))

class activation_module_low(layers.Layer):
    
    def __init__(self):
        
        super().__init__()
        self.w = self.add_weight(
        name='w',
        shape=(1,),
        initializer = 'random_normal',
        trainable = True
        )


    def call(self, x, inverse):
        
        if inverse == 0:

            t = tf.concat([[[0., 0.]], [[self.w[0], 0.]]], 0)        
            return x + tf.linalg.matvec(t, tf.math.tanh(x))
        
        else: 
            
            t = tf.concat([[[0., 0.]], [[-self.w[0], 0.]]], 0)        
            return x + tf.linalg.matvec(t, tf.math.tanh(x))



class SympNet(keras.Model):
    
    def __init__(self, N_layers, N_sub):
        
        super().__init__()
        
        
        self.Modules = []
        
        for i in range(N_layers):
            
            if i%2==0:
                self.Modules.append(activation_module_up())
                
            else:
                self.Modules.append(activation_module_low())
                
            for j in range(N_sub):
                
                if j==0:
                    self.Modules.append(bias())
                
                if j%2==0:
                    self.Modules.append(linear_module_up())

                else:
                    self.Modules.append(linear_module_low())
        
        

    def call(self, input_tensor):
        
        boom = self.Modules[0](input_tensor, 0)
        
        for i in range(1,len(self.Modules)):
            boom = self.Modules[i](boom, 0)
        
        boom = boom*tf.constant([1., -1.])        
        
        for i in range(1, len(self.Modules)):
            boom = self.Modules[-i](boom, 1)
        
        boom = self.Modules[0](boom, 1)*tf.constant([1., -1.])
    
        return boom