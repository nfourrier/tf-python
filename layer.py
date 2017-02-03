import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Layer(object):
    def __init__(self,*args):
        self._params = list(args)
        self.type = list(args)[0]
        self.number = list(args)[1]
        self.weights = None
        self.biases = None
        self.out = None
        self.scope = '{}-{}'.format(self.number,self.type)
        print(*args)
        self.setup(*args[2:]) # set attr up 
    def setup(self, *args): pass

class dropout_layer(Layer):
    def setup(self, stride):
        self.stride = stride

class connected_layer(Layer):
    def setup(self, stride):
        self.stride = stride

class maxpool_layer(Layer):
    def setup(self, size, stride, padding, inp):
        # self.scope = str(idx)+'-maxpool'
        self.inp = inp[0]
        self.out = tf.nn.max_pool(
            self.inp, padding = 'SAME',
            ksize = [1] + [size]*2 + [1], 
            strides = [1] + [stride]*2 + [1],
            name = self.scope
        )
        return self.out

class fullyconnected_layer(Layer):
    def setup(self,N_output,activation,inp):
        self.inp = inp[0]
        self.trainable = False
        input_size = int(np.prod(self.inp.get_shape()[1:]))
        
        self.weights = tf.Variable(
                         tf.truncated_normal([input_size,N_output],stddev=0.1),
                         trainable=self.trainable,
                         name=self.scope+'/kernel')
        self.biases = tf.Variable(
                         tf.constant(0.1, shape=[N_output]),
                         trainable=self.trainable,
                         name=self.scope+'/biases')
        x = tf.reshape(self.inp, [-1, input_size])
        self.out = tf.nn.bias_add(tf.matmul(x, self.weights), self.biases)
        if(activation=='relu'):
            temp = tf.nn.bias_add(tf.matmul(x, self.weights), self.biases)
            self.out = tf.nn.relu(temp,name=self.scope+'-out')
        elif(activation=='leaky'):
            alpha = 0.1
            temp = tf.nn.bias_add(tf.matmul(x, self.weights), self.biases)
            self.out = tf.maximum(alpha * temp,temp,name = self.scope+'-out')
        elif(activation=='softmax'):
            temp = tf.nn.bias_add(tf.matmul(x, self.weights), self.biases)
            self.out = tf.nn.softmax(temp)
        else:
            temp = tf.nn.bias_add(tf.matmul(x, self.weights), self.biases,name= self.scope+'-out')
        return self.out


class convolutional_layer(Layer):
    def setup(self,size,channels,filters,stride,padding,batch_norm,activation,inp):
        self.inp = inp[0]
        self.trainable = False
        channels = self.inp.get_shape()[3]
        self.weights = tf.Variable(
                         tf.truncated_normal([size, size, int(channels), filters],stddev=0.1),
                         trainable=self.trainable,
                         name=self.scope+'/kernel')
        self.biases = tf.Variable(
                         tf.constant(0.1, shape=[filters]),
                         trainable=self.trainable,
                         name=self.scope+'/biases')
        padder = [[padding, padding]] * 2
        temp = tf.pad(self.inp, [[0, 0]] + padder + [[0, 0]])
        temp = tf.nn.conv2d(temp, self.weights, padding = 'VALID', name = self.scope, strides = [1] + [stride] * 2 + [1])
        if(batch_norm): 
            temp = self.batch_normalization(self.weights, self.biases, temp, filters, self.scope, self.trainable)        
        
        if(activation=='relu'):
            temp = tf.nn.bias_add(temp, self.biases)
            self.out = tf.nn.relu(temp,name=self.scope+'-out')
        elif(activation=='leaky'):
            alpha = 0.1
            temp = tf.nn.bias_add(temp, self.biases)
            self.out = tf.maximum(alpha * temp,temp,name = self.scope+'-out')
        else:
            self.out = tf.nn.bias_add(temp, self.biases, name=self.scope+'-out')
        return self.out

    def batch_normalization(self,weights, biases, inp, filters, scope, trainable):
        args = [0., 1e-2, filters]
        moving_mean = np.random.normal(*args)
        moving_variance = np.random.normal(*args)
        gamma = np.random.normal(*args)

        moving_mean = tf.constant_initializer(moving_mean)
        moving_variance = tf.constant_initializer(moving_variance)
        gamma = tf.constant_initializer(gamma)

        args = dict({
            'center' : False, 'scale' : True,
            'epsilon': 1e-5, 'scope' : scope,
            'is_training': False
            })
        v = tf.__version__.split('.')[1]
        if int(v) < 12: key = 'initializers'
        else: key = 'param_initializers'
        w = {}
        w['moving_mean'] = moving_mean
        w['moving_variance'] = moving_variance
        w['gamma'] = gamma
        w['kernel'] = weights
        w['biases'] = biases

        args.update({key : w})
        return slim.batch_norm(inp, **args)

class leaky_layer(Layer):    
    def setup(self,inp):
        # self.scope = str(idx)+'-leaky'
        self.inp = inp[0]
        alpha = 0.1
        self.out = tf.maximum(alpha * self.inp,self.inp,name = self.scope+'-out')
        return self.out

class relu_layer(Layer):
    def setup(self,inp):
        self.inp = inp[0]
        self.out = tf.nn.relu(self.inp)
        return self.out

class route_layer(Layer):
    def setup(self, *args):
        # self.scope = str(idx+'-route')
        # self.inp = *args
        self.out = tf.concat(*args,3,name=self.scope+'-out')
        return self.out


class reorg_layer(Layer):
    def setup(self, stride, inp):
        print('la',stride,inp)
        self.stride = stride
        self.inp =inp[0]
        self. out = tf.extract_image_patches(self.inp, [1,stride,stride,1], [1,stride,stride,1], [1,1,1,1], 'VALID',name=self.scope+'-out')
        return self.out

layers = {
    'dropout': dropout_layer,
    'connected': connected_layer,
    'maxpool': maxpool_layer,
    'fullyconnected': fullyconnected_layer,
    'convolutional': convolutional_layer,
    'leaky': leaky_layer,
    'relu': relu_layer,
    'route': route_layer,
    'reorg': reorg_layer,
}        


def create_layer(ltype, num, *args):
    op_class = layers.get(ltype, Layer)
    return op_class(ltype, num, *args)