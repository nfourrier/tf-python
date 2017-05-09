import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Layer(object):
    '''
    Abstract parent class for all layers.
    To create a new class, say sm_layer:

        class sm_layer(Layer):
            def setup(self, arg1, arg2):
                self.out = arg1 + arg2
                return self.out
    '''
    def __init__(self,*args):
        self._params = list(args)
        self.type = list(args)[0]
        self.number = list(args)[1]
        self.dim = list(args)[2]
        self.weights = None
        self.biases = None
        self.out = None
        self.scope = '{}-{}'.format(self.number,self.type)
        try:
            self.setup(*args[3:]) # set attr up 
        except Exception as e:
            print('\t Error in layer {} with type {} \n\t Input parameters: {}'.format(self.number,self.type,*args[2:]))
            print(e)
            raise
    def setup(self, *args): pass


class dropout_layer(Layer):
    def setup(self, stride):
        self.stride = stride

class connected_layer(Layer):
    def setup(self, stride):
        self.stride = stride

class maxpool_layer(Layer):
    def setup(self, inp, size, stride, padding):
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
    def setup(self,inp,N_output,activation):
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
        
        self.out = tf.nn.bias_add(tf.matmul(x, self.weights), self.biases,name= self.scope+'-out')
        return self.out

class deconvolutional_layer_same(Layer):
    def setup(self, inp, size, stride, padding, output_shape,activation,trainable):
        self.inp = inp[0]
        
        add_layers = False
        if(output_shape[0]<0):
            target_layer = inp[1]
            out_features = target_layer.get_shape()[3].value 
            output_shape = tf.shape(target_layer)
            add_layers = True
        else:
            out_features = output_shape[2]
            input_shape = tf.shape(self.inp)
            output_shape = tf.stack([input_shape[0]]+output_shape)


        in_features = self.inp.get_shape()[3].value
        
        
        self.weights = tf.Variable(
                         tf.truncated_normal([size, size, out_features, in_features],stddev=0.1),
                         trainable=trainable,
                         name=self.scope+'/kernel')
        self.biases = tf.Variable(
                         tf.constant(0.1, shape=[out_features]),
                         trainable=trainable,
                         name=self.scope+'/biases')
        padder = [[padding, padding]] * 2
        temp = tf.pad(self.inp, [[0, 0]] + padder + [[0, 0]])
        deconv = tf.nn.conv2d_transpose(temp, self.weights, output_shape,
                                        strides=[1] + [stride] * 2 + [1], 
                                        padding='SAME') ## SAME
        out = tf.nn.bias_add(deconv, self.biases)
        # print(self.out)

        if(add_layers):
            out = tf.add(out,target_layer)

        self.out = tf.identity(out,name=self.scope+'/deconv')
        return self.out

class deconvolutional_layer(Layer):
    def setup(self, inp, size, stride, padding, output_shape,activation,trainable):
        self.inp = inp[0]
        
        add_layers = False
        if(output_shape[0]<0):
            target_layer = inp[1]
            out_features = target_layer.get_shape()[3].value 
            output_shape = tf.shape(target_layer)
            add_layers = True
        else:
            out_features = output_shape[2]
            input_shape = tf.shape(self.inp)
            output_shape = tf.stack([input_shape[0]]+output_shape)


        in_features = self.inp.get_shape()[3].value
        
        
        self.weights = tf.Variable(
                         tf.truncated_normal([size, size, out_features, in_features],stddev=0.1),
                         trainable=trainable,
                         name=self.scope+'/kernel')
        self.biases = tf.Variable(
                         tf.constant(0.1, shape=[out_features]),
                         trainable=trainable,
                         name=self.scope+'/biases')
        padder = [[padding, padding]] * 2
        temp = tf.pad(self.inp, [[0, 0]] + padder + [[0, 0]])
        pd2 = int((inp_shp[1]-1)*stride+size-out_shp[1])
        deconv = tf.nn.conv2d_transpose(temp, self.weights, output_shape,
                                        strides=[1] + [stride] * 2 + [1], 
                                        padding='VALID') 
        self.out = tf.nn.bias_add(deconv, self.biases)

        if(add_layers):
            self.out = tf.add(self.out,target_layer)


        return self.out

class convolutional_layer(Layer):
    def setup(self,inp,size,channels,filters,channels,stride,padding,batch_norm,activation):
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


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,[tf.div(shape[0], dilation), -1, shape[2]])


class res_block(Layer):
    def setup(self,inp,kernel_size,out_size,stride,dilation,batch_norm,activation,trainable):
        
        if(isinstance(dilation,type(int(1)))):
            dilation = [dilation]


        self.inp = inp[0]
        self.trainable = bool(trainable)
        self.activation = activation


        inp = self.inp
        out = 0
        # print('res',inp)
        for r in dilation:
            print(inp)
            conv_filter = self.single_aconv1d(inp, size=7, dilation=r, activation='tanh',scope=self.scope + '_aconv_{}_{}'.format('filter',r))
            # print(conv_filter)
            # exit()
            conv_gate = self.single_aconv1d(inp, size=7, dilation=r, activation='sigmoid',scope=self.scope + '_aconv_{}_{}'.format('gate',r))
            # print(conv_gate)
            res = conv_filter * conv_gate
            # print(res)
            res =  self.single_conv1d(res, size=1, activation='tanh',scope=self.scope + '_conv_{}_{}'.format('out',r))
            # exit()
            out += res

            inp = inp + res

        self.out = out
        # exit()
        print(self.out)
        # exit()
        return self.out

    def single_aconv1d(self,inp, size, dilation, activation, scope):
        args = (
                [inp],#inp
                size,#kernel_size
                128,#out_size
                1,#stride
                dilation,#dilation
                False,#batch_norm
                activation,#activation
                self.trainable,#trainable
                scope#scope
                )
        self.out = create_layer('causal_convolutional_1d',self.number,self.dim,
                *args
                ).out
        return self.out
    # inp,kernel_size,in_size,out_size,stride,padding,kernel,batch_norm,activation,trainable,scope=''
    def single_conv1d(self,inp, size, activation, scope):
        args = (
                [inp],#inp
                size,#kernel_size
                0,#in_size
                128,#out_size
                1,#stride
                0,#padding
                'identity',#kernel
                False,#batch_norm
                activation,#activation
                self.trainable,#trainable
                scope#scope
                )
        self.out = create_layer('convolutional_1d',self.number,self.dim,
                *args
                ).out
        return self.out

class causal_convolutional_1d(Layer):
    def setup(self,inp,kernel_size,in_size,out_size,stride,padding,kernel,batch_norm,activation,trainable):
        self.inp = inp[0]
        
        in_size = self.inp.get_shape()[2]
        
        self.trainable = bool(trainable)

        self.weights = tf.Variable(
                         tf.truncated_normal([kernel_size, int(in_size), out_size],stddev=0.1),
                         # tf.truncated_normal([1, kernel_size, int(in_size), out_size],stddev=0.1), ######### 1d case treated as 2d
                         trainable=trainable,
                         name=self.scope+'/kernel')
        # self.biases = tf.Variable(
        #                  tf.constant(0.1, shape=[out_size]),
        #                  trainable=trainable,
        #                  name=self.scope+'/biases')

        
        
        padding = [[0, 0], [(kernel_size - 1) * dilation + int((kernel_size-1)/2)*dilation, (kernel_size - 1) * dilation + int((kernel_size-1)/2)*dilation], [0, 0]]
        
        padded = tf.pad(self.inp, padding)

        if dilation > 1:
            transformed = time_to_batch(padded, dilation)
            conv = tf.nn.conv1d(transformed, self.weights, stride=1, padding='VALID')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(padded, self.weights, stride=1, padding='VALID')
        # Remove excess elements at the end.
        temp = tf.slice(restored,
                          [0, (kernel_size - 1) * dilation, 0],
                          [-1, tf.shape(self.inp)[1], -1])
                          # [-1, -1, -1])

        alpha = -1
        beta = -1
        self.out = create_layer('activation',self.number,self.dim,[temp],activation,alpha,beta,self.scope).out
        return self.out

class route_layer(Layer):
    def setup(self, *args):
        self.out = tf.concat(*args,3,name=self.scope+'-out')
        return self.out


class reorg_layer(Layer):
    def setup(self, inp, stride):
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
    'deconvolutional': deconvolutional_layer,
    'causal_convolutional_1d': causal_convolutional_1d,
    'res_block': res_block,
    'leaky': leaky_layer,
    'relu': relu_layer,
    'route': route_layer,
    'reorg': reorg_layer,
}        


def create_layer(ltype, num, *args):
    op_class = layers.get(ltype, Layer)
    return op_class(ltype, num, *args)


    

def main():
    print('Not implemented')

if __name__ == '__main__':
    main()