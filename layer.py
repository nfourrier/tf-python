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
    
    def __str__(self):
        return "Layer {0}: {1} \n\t IN: {2}\n\tOUT: {3}".format(self.number,self.type,self.inp,self.out)
  


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
        '''
        Args:
            inp: list of tf layers
                the second element of inp is the target layer used to determine the shape of the deconv output
        input_shape: `(samples, rows, cols, channels)`
        output_shape: Output shape of the transposed convolution operation.
        tuple of integers (nb_samples, nb_output_rows, nb_output_cols, nb_filter)
        
        Formula for calculation of the output shape [1], [2]:
        o = s (i - 1) + a + k - 2p, \quad a \in \{0, \ldots, s - 1\}
        where:
            - i - input size (rows or cols),
            - k - kernel size (nb_filter),
            - s - stride (subsample for rows or cols respectively),
            - p - padding size,
            - a - user-specified quantity used to distinguish between
            -     the s different possible output sizes.
        Because a is not specified explicitly and Theano and Tensorflow
        use different values, it is better to use a dummy input and observe
        the actual output shape of a layer as specified in the examples.
        # References
        [1] [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285 "arXiv:1603.07285v1 [stat.ML]")
        [2] [Transposed convolution arithmetic](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic)
        [3] [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
        '''
        self.inp = inp[0]
        
        input_shape = tf.shape(self.inp)
        input_shape = self.inp.get_shape()[1].value
        input_shape = [self.inp.get_shape()[x].value for x in range(4)]

        output_shape = [None] + output_shape
        tf_output_shape = tf.stack([tf.shape(self.inp)[0]]+output_shape[1:])
        
        in_features = input_shape[3]
        out_features = output_shape[3]        
        
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
        
        pad_total = int((input_shape[1]-1)*stride+size-output_shape[1])
        pad_before = int(pad_total/2)
        pad_after = pad_total - pad_before
        tf_output_shape = tf.stack([tf.shape(temp)[0]]+[output_shape[1]+pad_total,output_shape[2]+pad_total,output_shape[3]])

        deconv = tf.nn.conv2d_transpose(temp, self.weights, tf_output_shape,
                                        strides=[1] + [stride] * 2 + [1], 
                                        padding='VALID') ## SAME
        deconv = tf.slice(deconv,tf.stack([0,pad_before,pad_before,0]),tf.stack([-1,output_shape[1],output_shape[2],-1]))

        temp = tf.nn.bias_add(deconv, self.biases)

        if(len(inp) > 1):
            temp = tf.add(temp,inp[1])
        alpha = -1
        beta = -1
        self.out = create_layer('activation',self.number,self.dim,[temp],activation,alpha,beta,self.scope).out

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


class convolutional_1d_layer(Layer):
    # def setup(self,size,channels,filters,stride,padding,kernel,batch_norm,activation,trainable,inp):
    def setup(self,inp,kernel_size,in_size,out_size,stride,padding,kernel,batch_norm,activation,trainable,scope=''):
        '''
        PADDING = VALID OR SAME
        "VALID" = without padding:
        inputs:         1  2  3  4  5  6  7  8  9  10 11 (12 13)
                        |________________|                dropped
                                        |_________________|

        "SAME" = with zero padding:
                    pad|                                      |pad
        inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
                    |________________|
                                   |_________________|
                                                  |________________|                                        
        '''
        # self.scope = str(self.number)+'-convolutional'

                if(scope!=''):
            self.scope=scope
        
        self.inp = inp[0]
        self.trainable = bool(trainable)

        in_size = self.inp.get_shape()[2]

        self.weights = tf.Variable(
                         tf.truncated_normal([kernel_size, int(in_size), out_size],stddev=0.1),
                         trainable=trainable,
                         name=self.scope+'/kernel')
        self.biases = tf.Variable(
                         tf.constant(0., shape=[out_size]),
                         trainable=trainable,
                         name=self.scope+'/biases')
        
        padder = [[padding, padding]] * 1
        
        temp = tf.pad(self.inp, [[0, 0]] + padder + [[0, 0]])
        
        temp = tf.nn.conv1d(temp, self.weights, padding = 'VALID', name = self.scope, stride = stride)

        if(batch_norm): 
            temp = self.batch_normalization(self.weights, self.biases, temp, out_size, self.scope, self.trainable)        
        
        temp = tf.nn.bias_add(temp, self.biases)
        alpha = -1
        beta = -1
        self.out = create_layer('activation',self.number,self.dim,[temp],activation,alpha,beta,self.scope).out



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
        # if int(v) < 12: key = 'initializers'
        # else: key = 'param_initializers'
        key = 'param_initializers'
        w = {}
        w['moving_mean'] = moving_mean
        w['moving_variance'] = moving_variance
        w['gamma'] = gamma
        w['kernel'] = weights
        w['biases'] = biases

        args.update({key : w})
        return slim.batch_norm(inp, **args)

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

class normalize_layer(Layer):        
    def setup(self,inp,norm=2):
        self.inp = inp[0]
        self.out = self.inp / tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.pow(tensor,norm),axis=1)),1)
        return self.out

class cross_map_lrn(Layer):
    def setup(self,inp, size=5, bias=1.0, alpha=1.0, beta=0.5):
        self.inp = inp[0]
        padding = int(size/2)
        tensor_pad = tf.pad(self.inp, [[0, 0]] + [[0,0]]*2 + [[padding, padding]])

        tensor_pad = tf.expand_dims(tensor_pad,4)

        squared = tf.square(tensor_pad)
        in_channels = tensor_pad.get_shape().as_list()[3]
        kernel = tf.constant(1.0, shape=[1, 1, size, 1,1])
        squared_sum = tf.nn.conv3d(squared,
                                           kernel,
                                           [1, 1, 1, 1,1],
                                           padding='VALID')[:,:,:,:,0]
        bias = tf.constant(bias, dtype=tf.float32)
        alpha = tf.constant(alpha, dtype=tf.float32)
        beta = tf.constant(beta, dtype=tf.float32)
        self.out = tensor / ((bias + alpha/size * squared_sum) ** beta)
        return self.out

class lp_pool_layer(Layer):
    def setup(self,inp, pnorm, kH, kW, dH, dW, padding, name):
        self.inp = inp[0]
        if pnorm == 2:
            pwr = tf.square(self.inp)
        else:
            pwr = tf.pow(self.inp, pnorm)
          
        subsamp = tf.nn.avg_pool(pwr,
                              ksize=[1, kH, kW, 1],
                              strides=[1, dH, dW, 1],
                              padding='VALID')
        subsamp_sum = tf.multiply(subsamp, kH*kW)
        
        if pnorm == 2:
            self.out = tf.sqrt(subsamp_sum)
        else:
            self.out = tf.pow(subsamp_sum, 1/pnorm)
        return self.out        
  

class batch_norm(Layer):
    def setup(self, inp, trainable):
        """
        Batch normalization on convolutional maps.
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Variable, true indicates training phase
            scope:       string, variable scope
            affn:      whether to affn-transform outputs
        Return:
            normed:      batch-normalized maps
        Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177
        """
        self.inp = inp[0]
        inp_type = self.inp.dtype
        scope = 'batch_norm'
        phase_train = tf.convert_to_tensor(trainable, dtype=tf.bool)
        n_out = int(self.inp.get_shape()[3])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=inp_type),
                           name=scope+'/beta', trainable=True, dtype=inp_type)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=inp_type),
                            name=scope+'/gamma', trainable=True, dtype=inp_type)
      
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        

        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = control_flow_ops.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (ema.average(batch_mean), ema.average(batch_var)))

        self.out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return self.out


class inception(Layer):   
    def setup(self,inp, inSize, ks, o1s, o2s1, o2s2, o3s1, o3s2, o4s1, o4s2, o4s3, poolType, name, 
                  phase_train=True, use_batch_norm=True, weight_decay=0.0):
      
        print('name = ', name)
        print('inputSize = ', inSize)
        print('kernelSize = {3,5}')
        print('kernelStride = {%d,%d}' % (ks,ks))
        print('outputSize = {%d,%d}' % (o2s2,o3s2))
        print('reduceSize = {%d,%d,%d,%d}' % (o2s1,o3s1,o4s2,o1s))
        print('pooling = {%s, %d, %d, %d, %d}' % (poolType, o4s1, o4s1, o4s3, o4s3))
        print("conv for pooling reduceSize = {%d}" % o4s2)
        if (o4s2>0):
            o4 = o4s2
        else:
            o4 = inSize
        print('outputSize = ', o1s+o2s2+o3s2+o4)
        print()
        
        net = []
        var_all = {}
        net_all = {}

        with tf.variable_scope(name):
            
            with tf.variable_scope('branch2_3x3'):
                if o2s1>0:
                    var_all['branch2_3x3'] = {}
                    # conv3a,var = conv(inp, inSize, o2s1, 1, 1, 1, 1, 0, 'conv1x1_tmp', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                    # if(name=='incept3cd'):
                    #     conv3a,var = full_conv2(inp, inSize, o2s1, 1, 1, 1, 1, 0, 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)    
                    #     # inp = conv3a 
                    # else:
                    conv3a,var = full_conv(inp, inSize, o2s1, 1, 1, 1, 1, 0, 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                    var_all['branch2_3x3']['conv1x1'] = var
                    conv3,var = full_conv(conv3a, o2s1, o2s2, 3, 3, ks, ks, 1, 'conv3x3', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                    var_all['branch2_3x3']['conv3x3'] = var
                    net.append(conv3)
                    net_all['branch2_3x3'] = conv3
                    # var_all['branch2_3x3'] = var
          
            with tf.variable_scope('branch3_5x5'):
                if o3s1>0:
                    var_all['branch3_5x5'] = {}
                    conv5a,var = full_conv(inp, inSize, o3s1, 1, 1, 1, 1, 0, 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                    var_all['branch3_5x5']['conv1x1'] = var
                    conv5,var = full_conv(conv5a, o3s1, o3s2, 5, 5, ks, ks, 2, 'conv5x5', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                    var_all['branch3_5x5']['conv5x5'] = var
                    net.append(conv5)
                    net_all['branch3_5x5'] = conv5

            with tf.variable_scope('branch4_pool'):
                if(o4s1==1):
                    pad = 0
                elif(o4s1==3):
                    pad = 1
                elif(o4s1==5):
                    pad = 2
                else:
                    raise ValueError('pooling padding not ready, max pool can be done with size 1,3,5 "%s"' % poolType)
                if poolType=='MAX':
                    pool,_ = mpool(inp, o4s1, o4s1, o4s3, o4s3, pad, 'pool')
                    pad = 0
                elif poolType=='L2':
                    pool,_ = lppool(inp, 2, o4s1, o4s1, o4s3, o4s3, 0, 'pool')
                else:
                    raise ValueError('Invalid pooling type "%s"' % poolType)
                # print(pool)
                if o4s2>0:
                    var_all['branch4_pool'] = {}
                    # if(name=='incept3b'):
                    #     pool_conv,var = full_conv2(pool, inSize, o4s2, 1, 1, 1, 1, 0, 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                    # else:
                    pool_conv,var = full_conv(pool, inSize, o4s2, 1, 1, 1, 1, 0, 'conv1x1', phase_train=phase_train, use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                    var_all['branch4_pool']['conv1x1'] = var
                else:
                    pool_conv = pool
                # print(pool_conv)
                if poolType=='L2':
                    pool_conv = tf.pad(pool_conv, [[0, 0]] + [[pad,pad]]*2 + [[0, 0]])
                    # print(pool_conv)
                    # exit()
                net.append(pool_conv)
                net_all['branch4_pool'] = pool_conv
                net_all['branch4_pool2'] = pool
            # exit()
        return net_all,var_all


layers = {
    'dropout': dropout_layer,
    'connected': connected_layer,
    'maxpool': maxpool_layer,
    'fullyconnected': fullyconnected_layer,
    'convolutional': convolutional_layer,
    'deconvolutional': deconvolutional_layer,
    'causal_convolutional_1d': causal_convolutional_1d,
    'convolutional_1d': convolutional_1d_layer,
    'res_block': res_block,
    'leaky': leaky_layer,
    'relu': relu_layer,
    'route': route_layer,
    'reorg': reorg_layer,
}        


##########################################################      
################## ACTIVATIONS FUNCTIONS #################      
def identity(inp,*args):        
    out = inp       
    return out      
def softmax(inp,*args):     
    out = tf.nn.softmax(inp)        
    return out      
def leaky(inp,alpha=0.1,*args):     
    if(alpha==-1):      
        alpha = 0.1     
    out = tf.maximum(alpha * inp,inp)       
    return out      
def relu(inp,*args):        
    out = tf.nn.relu(inp)       
    return out      
def tanh(inp,*args):        
    out = tf.tanh(inp)      
    return out      
def sigmoid(inp,*args):     
    out = tf.sigmoid(inp)       
    return out      
def softsign(inp,*args):        
    out = tf.nn.softsign(x)     
    return out      
def elu(inp,alpha,*args):       
    out = tf.nn.elu(x)      
    if alpha == -1:     
        alpha = 1       
        
    if alpha != 1:      
        out = tf.where(inp > 0, out, alpha * out)       
    return out      
def softplus(inp,*args):            
    out = tf.nn.softplus(inp)       
    return out      
        
        
activation_functions = {        
        'identity': identity,       
        'linear': identity,     
        'softmax': softmax,     
        'relu': relu,       
        'leaky': leaky,     
        'elu': elu,     
        'softplus': softplus,       
        'softsign': softsign,       
        'tanh': tanh,       
        'sigmoid': sigmoid,     
        }       
##########################################################



def create_layer(ltype, num, *args):
    op_class = layers.get(ltype, Layer)
    return op_class(ltype, num, *args)


    

def main():
    print('Not implemented')

if __name__ == '__main__':
    main()