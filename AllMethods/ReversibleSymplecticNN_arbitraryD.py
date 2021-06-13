import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

########################### Polynomial Henon Non-Reversible ###################

class HNR(layers.Layer):
    
    def __init__(self):
        super().__init__()

    
    def build(input_dimension):
        
        self.w = self.add_weight(
        name='w',
        shape=(5,),
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        trainable = True
        )

    def call(self, x):
        
        
        L0 = tf.concat([[[1., 0.],[0., 0.]]], 0)
        L10 = tf.concat([[[0., 0.],[0., -1]]], 0)
        
        L1 = tf.concat([[[0., 0.],[0., self.w[0]]]], 0)
        L2 = tf.concat([[[0., 0.],[0., self.w[1]]]], 0)
        L3 = tf.concat([[[0., 0.],[0., self.w[2]]]], 0)
        L4 = tf.concat([[[0., 0.],[0., self.w[3]]]], 0)
        #L5 = tf.concat([[[0., 0.],[0., self.w[4]]]], 0)
                

        x̄ = tf.reverse(x, [1])
        b = tf.constant([[0., 1.]])

        return tf.linalg.matvec(L0, x̄) + tf.linalg.matvec(L10, x̄) + tf.linalg.matvec(L1, b) + tf.linalg.matvec(L2, x) + tf.linalg.matvec(L3, x)*x + tf.linalg.matvec(L4, x)*x*x #+ tf.linalg.matvec(L5, x)*x*x*x

class HenonNR(keras.Model):
    
    def __init__(self, N):
        
        super().__init__()
        
        self.Hs = []
        for i in range(N):
            self.Hs.append(HNR())


    def call(self, input_tensor):
    
        boom = input_tensor
        for i in range(len(self.Hs)):
            boom = self.Hs[i](boom)

        return boom


############################## Polynomial Henon Reversible ####################
        
class HR(layers.Layer):
    
    def __init__(self):
        
        super().__init__()
        self.w = self.add_weight(
        name='w',
        shape=(5,),
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        trainable = True
        )

    def call(self, x, inverse):
        
        if inverse == 0:
        
            L0 = tf.concat([[[1., 0.],[0., 0.]]], 0)
            L10 = tf.concat([[[0., 0.],[0., -1]]], 0)

            L1 = tf.concat([[[0., 0.],[0., self.w[0]]]], 0)
            L2 = tf.concat([[[0., 0.],[0., self.w[1]]]], 0)
            L3 = tf.concat([[[0., 0.],[0., self.w[2]]]], 0)
            L4 = tf.concat([[[0., 0.],[0., self.w[3]]]], 0)
            #L5 = tf.concat([[[0., 0.],[0., self.w[4]]]], 0)


            x̄ = tf.reverse(x, [1])
            b = tf.constant([[0., 1.]])

            return tf.linalg.matvec(L0, x̄) + tf.linalg.matvec(L10, x̄) + tf.linalg.matvec(L1, b) + tf.linalg.matvec(L2, x) + tf.linalg.matvec(L3, x)*x + tf.linalg.matvec(L4, x)*x*x #+ tf.linalg.matvec(L5, x)*x*x*x
        
        else:
            
            L0 = tf.concat([[[0., 0.],[0., 1.]]], 0)
            L10 = tf.concat([[[-1., 0.],[0., 0.]]], 0)

            L1 = tf.concat([[[self.w[0], 0.],[0., 0.]]], 0)
            L2 = tf.concat([[[self.w[1], 0.],[0., 0.]]], 0)
            L3 = tf.concat([[[self.w[2], 0.],[0., 0.]]], 0)
            L4 = tf.concat([[[self.w[3], 0.],[0., 0.]]], 0)
            #L5 = tf.concat([[[self.w[4], 0.],[0., 0.]]], 0)
            
            
            x̄ = tf.reverse(x, [1])
            b = tf.constant([[1., 0.]])
            
            return tf.linalg.matvec(L0, x̄) + tf.linalg.matvec(L10, x̄) + tf.linalg.matvec(L1, b) + tf.linalg.matvec(L2, x) + tf.linalg.matvec(L3, x)*x + tf.linalg.matvec(L4, x)*x*x #+ tf.linalg.matvec(L5, x)*x*x*x

class HenonR(keras.Model):
    
    def __init__(self, N):
        
        super().__init__()
        
        self.Hs = []
        for i in range(N):
            self.Hs.append(HR())


    def call(self, input_tensor):
    
        boom = input_tensor
        for i in range(len(self.Hs)):
            boom = self.Hs[i](boom, 0)
            
        boom = boom*tf.constant([1., -1.])
        
        for i in range(1, len(self.Hs)):
            boom = self.Hs[-i](boom, 1)

        boom = self.Hs[0](boom, 1)*tf.constant([1., -1.])
        
        return boom

###############################################################################
###############################################################################
###############################################################################

class linear_module_up(layers.Layer):
    
    def __init__(self, bias):
        
        super().__init__()
        self.w = self.add_weight(
        name='w',
        shape=(1,),
        initializer = 'random_normal',
        trainable = True
        )
    
        if bias==1:
            self.b = self.add_weight(
            name='b',
            shape=(2,),
            initializer = 'random_normal',
            trainable = True
            )
            
        else:
            self.b = tf.constant([0., 0.])

    def call(self, x):
        
        L = tf.concat([[[1., self.w[0]]],[[0., 1.]]], 0)
        return tf.linalg.matvec(L, x) + self.b


class linear_module_low(layers.Layer):
    
    def __init__(self, bias):
        
        super().__init__()
        self.w = self.add_weight(
        name='w',
        shape=(1,),
        initializer = 'random_normal',
        trainable = True
        )

        if bias==1:
            self.b = self.add_weight(
            name='b',
            shape=(2,),
            initializer = 'random_normal',
            trainable = True
            )
            
        else:
            self.b = tf.constant([0., 0.])

    def call(self, x):
        
        L = tf.concat([[[1., 0.]],[[self.w[0], 1.]]], 0)
        return tf.linalg.matvec(L,x) + self.b


    

class activation_module_up(layers.Layer):
    
    def __init__(self):
        
        super().__init__()
        self.w = self.add_weight(
        name='w',
        shape=(1,),
        initializer = 'random_normal',
        trainable = True
        )


    def call(self, x):        
        t = tf.concat([[[0., self.w[0]]], [[0.,0.]]], 0)
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


    def call(self, x):
        t = tf.concat([[[0., 0.]], [[self.w[0], 0.]]], 0)        
        return x + tf.linalg.matvec(t, tf.math.tanh(x))



class SympNetNR(keras.Model):
    
    def __init__(self, N_layers, N_sub):
        
        super().__init__()
        
        
        self.Modules = []
        
        for i in range(N_layers):
            
            if i%2==0:
                self.Modules.append(activation_module_up())
                
            else:
                self.Modules.append(activation_module_low())
                
            for j in range(N_sub):
                
                if j==(N_sub-1):
                    bias = 1
                else:
                    bias = 0
                
                if j%2==0:
                    self.Modules.append(linear_module_up(bias))

                else:
                    self.Modules.append(linear_module_low(bias))
        
        

    def call(self, input_tensor):
        
        boom = self.Modules[0](input_tensor)
        
        for i in range(1,len(self.Modules)):
            boom = self.Modules[i](boom)
    
        return boom

###############################################################################
###############################################################################
###############################################################################


class linear_module_upR(layers.Layer):
    
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


class linear_module_lowR(layers.Layer):
    
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


            
class activation_module_upR(layers.Layer):
    
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

class activation_module_lowR(layers.Layer):
    
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



class SympNetR(keras.Model):
    
    def __init__(self, N_layers, N_sub):
        
        super().__init__()
        
        
        self.Modules = []
        
        for i in range(N_layers):
            
            if i%2==0:
                self.Modules.append(activation_module_upR())
                
            else:
                self.Modules.append(activation_module_lowR())
                
            for j in range(N_sub):
                
                if j==0:
                    self.Modules.append(bias())
                
                if j%2==0:
                    self.Modules.append(linear_module_upR())

                else:
                    self.Modules.append(linear_module_lowR())
        
        

    def call(self, input_tensor):
        
        boom = self.Modules[0](input_tensor, 0)
        
        for i in range(1,len(self.Modules)):
            boom = self.Modules[i](boom, 0)
        
        boom = boom*tf.constant([1., -1.])        
        
        for i in range(1, len(self.Modules)):
            boom = self.Modules[-i](boom, 1)
        
        boom = self.Modules[0](boom, 1)*tf.constant([1., -1.])
    
        return boom






###############################################################################
###############################################################################
###############################################################################
def SympNet(num_layers, num_sub, reversible):
    
    if reversible=='reversible':
    
        return SympNetR(num_layers, num_sub)
    
    if reversible=='non_reversible':
    
        return SympNetNR(num_layers, num_sub)

def Henon(num_layers, reversible):
    
     if reversible=='reversible':
    
        return HenonR(num_layers)
    
     if reversible=='non_reversible':
    
        return HenonNR(num_layers)