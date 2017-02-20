import tensorflow as tf


import dasakl.nn.layer as lay
class mmodel(object):
    def __init__(self,):
        self.name = 'yolo'


    def parse(self,cfg_file):
        from dasakl.nn.parser import get_layers
        layers = get_layers(cfg_file)
        return layers

    def get_input(self,dim0,dim1,dim2):

        self.inp = tf.placeholder(tf.float32,[None, dim0, dim1, dim2])
        return self.inp


    def set_input(self,dim):
        self.inp = tf.placeholder(tf.float32,[None]+dim)
        return self.inp


    def get_model(self,layers):
        inp = [self.inp]

        self._layers = {}
        self._variables = {}

        N_layer = 0
        for layer in layers:
            param = layer

            if(N_layer==0):
                param[-1] = [self.inp]
            else:
                param[-1] = [self.layers['{}_{}'.format(param[-1][idx][0],param[-1][idx][1])] for idx in range(len(param[-1]))]
            
            tf_layer = lay.create_layer(*param)
            self._layers['{}_{}'.format(param[0],param[1])] = tf_layer.out
            N_layer += 1
            print(param)
            if(param[0]=='fullyconnected'):
                self._layers['flatten_input'] = tf_layer.flatten[0]
                self._layers['flatten_output'] = tf_layer.flatten[1]
                # self._layers['flatten_output_transpose'] = tf_layer.flatten[1]
        
        # import pprint
        # pprint.pprint(self.layers)

        # for var in tf.global_variables(): 
        #     print(var.name,type(var))

        
        for var in tf.global_variables(): ### old version was tf.all_variables()
            # print(var.name)
            self._variables[var.name] = var

        return self.inp,self._layers['{}_{}'.format(param[0],param[1])],self._variables

class tf_model(object):
    def weight_variable(self,shape,name,trainable):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial,name=name,trainable=trainable)


    def bias_variable(self,shape,name,trainable):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial,name=name,trainable=trainable)

    def deconv_variable(self, shape,name,trainable):
        from math import ceil
        import numpy as np
        width = shape[0]
        height = shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([shape[0], shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(shape)
        for i in range(shape[2]):
            weights[:, :, i, i] = bilinear

        initial = tf.constant(value=weights,dtype=tf.float32)
        return tf.Variable(initial,name=name,trainable=trainable)




    def avg_pool(self,bottom, name):
        return tf.nn.avg_pool(bottom, ksize=2, strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self,bottom, name, kernel_size=2,strides=[1,2,2,1]):
        return tf.nn.max_pool(bottom, ksize=[1,kernel_size,kernel_size,1], strides=strides, padding='SAME', name=name)

    def lrn_layer(self,bottom,name,depth_radius=5,alpha=1,beta=0.1):
        return tf.nn.local_response_normalization(input, depth_radius=None, bias=None, alpha=None, beta=None, name=None)

    def conv_layer(self,bottom, in_channels, out_channels, name, kernel_size=3, strides=[1,1,1,1],padding='SAME',trainable=True):
        # with tf.variable_scope(name):
        # filter_size = 3
        biases = self.bias_variable([out_channels],name+'_bias',trainable)
        weights = self.weight_variable([kernel_size,kernel_size,in_channels,out_channels],name+'_weight',trainable)
        conv = tf.nn.conv2d(bottom, weights, strides, padding=padding)
        bias = tf.nn.bias_add(conv,biases)
        relu = tf.nn.relu(bias)
        print(name,bottom.get_shape(),relu.get_shape(),weights.get_shape())
        return relu

    def deconv_layer(self, bottom, out_features, output_shape, name, kernel_size=3, strides=[1,2,2,1],trainable=True):
        '''
            input_shape: `(samples, rows, cols, channels)`
            output_shape: Output shape of the transposed convolution operation.
                    tuple of integers (nb_samples, nb_output_rows, nb_output_cols, nb_filter)
                    Formula for calculation of the output shape [1], [2]:
                        o = s (i - 1) + a + k - 2p, \quad a \in \{0, \ldots, s - 1\}
                        where:
                            i - input size (rows or cols),
                            k - kernel size (nb_filter),
                            s - stride (subsample for rows or cols respectively),
                            p - padding size,
                            a - user-specified quantity used to distinguish between
                                the s different possible output sizes.
                     Because a is not specified explicitly and Theano and Tensorflow
                     use different values, it is better to use a dummy input and observe
                     the actual output shape of a layer as specified in the examples.
                     # References
                    [1] [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285 "arXiv:1603.07285v1 [stat.ML]")
                    [2] [Transposed convolution arithmetic](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic)
                    [3] [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
        '''
        in_features = bottom.get_shape()[3].value


        # if shape is None:
        #     # Compute shape out of Bottom
        #     in_shape = tf.shape(bottom)

        #     h = ((in_shape[1] - 1) * stride) + 1
        #     w = ((in_shape[2] - 1) * stride) + 1
        #     new_shape = [in_shape[0], h, w, num_classes]
        # else:
        # new_shape = [channels, width, height, N_classes]
        output_shape = tf.pack(output_shape)

        # logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
        weight_shape = [kernel_size, kernel_size, out_features, in_features]
        ##filter shape [height,width,output_channels,in_channels]

        # create
        # num_input = ksize * ksize * in_features / stride
        # stddev = (2 / num_input)**0.5

        # weights = self.deconv_variable(weight_shape,name+'_weight',trainable)
        weights = self.weight_variable(weight_shape,name+'_weight',trainable)
        biases = self.bias_variable([out_features],name+'_bias',trainable)
        # print(weights.get_shape())
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')
        deconv = tf.nn.bias_add(deconv, biases)
        print(' ',output_shape,weights.get_shape())
        print(name,bottom.get_shape(),deconv.get_shape(),weights.get_shape(),output_shape)
        return deconv

# deconv keras
# def deconv2d(x, kernel, output_shape, strides=(1, 1),
#              border_mode='valid',
#              dim_ordering=_IMAGE_DIM_ORDERING,
#              image_shape=None, filter_shape=None):
#     '''2D deconvolution (i.e. transposed convolution).
#     # Arguments
#         x: input tensor.
#         kernel: kernel tensor.
#         output_shape: 1D int tensor for the output shape.
#         strides: strides tuple.
#         border_mode: string, "same" or "valid".
#         dim_ordering: "tf" or "th".
#             Whether to use Theano or TensorFlow dimension ordering
#             for inputs/kernels/ouputs.
#     '''
#     if dim_ordering not in {'th', 'tf'}:
#         raise Exception('Unknown dim_ordering ' + str(dim_ordering))

#     x = _preprocess_conv2d_input(x, dim_ordering)
#     output_shape = _preprocess_deconv_output_shape(output_shape, dim_ordering)
#     kernel = _preprocess_conv2d_kernel(kernel, dim_ordering)
#     kernel = tf.transpose(kernel, (0, 1, 3, 2))
#     padding = _preprocess_border_mode(border_mode)
#     strides = (1,) + strides + (1,)

#     x = tf.nn.conv2d_transpose(x, kernel, output_shape, strides,
#                                padding=padding)
#     return _postprocess_conv2d_output(x, dim_ordering)




    def fc_layer(self,bottom, in_size, out_size, name, trainable=True):
        # with tf.variable_scope(name):
        biases = self.bias_variable([out_size],name+'_bias',trainable)
        weights = self.weight_variable([in_size,out_size],name+'_weight',trainable)
        x = tf.reshape(bottom, [-1, in_size])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        return fc

    # def save(self,session,filename,var_dict):

    # def load(self,)



class tf_vgg(tf_model):
    def __init__(self,N_classes):
        self.N_classes = N_classes
        if(N_classes==21):
            self.fc_size = 256
        else:
            self.fc_size = 4096
        self._layers = {}

    def get_training_model(self):
        layers = {}
        x = tf.placeholder(tf.float32, [None, 224, 224, 3])

        layers['conv1_1'] = self.conv_layer(x, 3, 64, "conv1_1",trainable=False)
        layers['conv1_2'] = self.conv_layer(layers['conv1_1'], 64, 64, "conv1_2",trainable=False)
        layers['pool1'] = self.max_pool(layers['conv1_2'], 'pool1')

        layers['conv2_1'] = self.conv_layer(layers['pool1'], 64, 128, "conv2_1",trainable=False)
        layers['conv2_2'] = self.conv_layer(layers['conv2_1'], 128, 128, "conv2_2",trainable=False)
        layers['pool2'] = self.max_pool(layers['conv2_2'], 'pool2')

        layers['conv3_1'] = self.conv_layer(layers['pool2'], 128, 256, "conv3_1",trainable=False)
        layers['conv3_2'] = self.conv_layer(layers['conv3_1'], 256, 256, "conv3_2",trainable=False)
        layers['conv3_3'] = self.conv_layer(layers['conv3_2'], 256, 256, "conv3_3",trainable=False)
        # layers['conv3_4'] = self.conv_layer(layers['conv3_3'], 256, 256, "conv3_4")
        layers['pool3'] = self.max_pool(layers['conv3_3'], 'pool3')

        layers['conv4_1'] = self.conv_layer(layers['pool3'], 256, 512, "conv4_1",trainable=False)
        layers['conv4_2'] = self.conv_layer(layers['conv4_1'], 512, 512, "conv4_2",trainable=False)
        layers['conv4_3'] = self.conv_layer(layers['conv4_2'], 512, 512, "conv4_3",trainable=False)
        # layers['conv4_4'] = self.conv_layer(layers['conv4_3'], 512, 512, "conv4_4")
        layers['pool4'] = self.max_pool(layers['conv4_3'], 'pool4')

        layers['conv5_1'] = self.conv_layer(layers['pool4'], 512, 512, "conv5_1",trainable=False)
        layers['conv5_2'] = self.conv_layer(layers['conv5_1'], 512, 512, "conv5_2",trainable=False)
        layers['conv5_3'] = self.conv_layer(layers['conv5_2'], 512, 512, "conv5_3",trainable=False)
        # layers['conv5_4'] = self.conv_layer(layers['conv5_3'], 512, 512, "conv5_4")
        layers['pool5'] = self.max_pool(layers['conv5_3'], 'pool5')


        layers['fc6'] = self.fc_layer(layers['pool5'], 25088, self.fc_size, "fc6",trainable=True)  # 25088 = ((224 / (2 ** 5)) ** 2) * 512
        layers['fc6'] = tf.nn.dropout(layers['fc6'], 0.5)

        layers['fc7'] = self.fc_layer(layers['fc6'], self.fc_size, self.fc_size, "fc7",trainable=True)
        layers['fc7'] = tf.nn.dropout(layers['fc7'], 0.5)

        layers['fc8'] = self.fc_layer(layers['fc7'], self.fc_size, self.N_classes, "fc8",trainable=True)
        y = layers['fc8']


        # layers['deconv'] = tf.nn.conv2d_transpose(layers['fc7'], weights, output_shape,
        #                                     strides=strides, padding='SAME')

        self._layers = layers
        params = {}
        for var in tf.all_variables():
            # print(var.name)
            params[var.name] = var
        self._variables = params
        return x, y,params


    @property
    def layers(self):
        return self._layers

    @property
    def variables(self):
        return self._variables

    def get_loss(self,y,y_):
        cross_entropy = -y_ * tf.log(tf.nn.softmax(y))
        return tf.reduce_mean(cross_entropy)

    def get_optimizer(self,loss,lr):
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=lr,
            initial_accumulator_value=0.1,
            use_locking=False,
            name='Adagrad')
        return optimizer.minimize(loss)


