import tensorflow as tf


class bl_layer(object):
    def __init__(self,name,type,params):
        self.name = name

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
        # return tf.get_variable(name="up_filter", initializer=init,
        #                        shape=weights.shape)



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

class tf_yo(tf_model):
    def __init__(self,N_classes):
        self.N_classes = 1000


    def get_training_model(self):
        x = tf.placeholder(tf.float32, [None, 224, 224, 3])

        conv1 = self.conv_layer(x, in_channels=3, out_channels=96, name='conv1', kernel_size=11, strides=[1,4,4,1])
        pool1 = self.max_pool(conv1, 'pool1', kernel_size=3,strides=[1,2,2,1])
        lrn1 = self.lrn_layer(pool1,name='lrn1',depth_radius=5,alpha=0.0001,beta=0.75)

        conv2 = self.conv_layer(lrn1, in_channels=96, out_channels=256, name='conv2', kernel_size=5, strides=[1,1,1,1])
        pool2 = self.max_pool(conv2, 'pool1', kernel_size=3,strides=[1,2,2,1])
        lrn2 = self.lrn_layer(pool2,name='lrn2',depth_radius=5,alpha=0.0001,beta=0.75)

        conv3 = self.conv_layer(lrn2, in_channels=256, out_channels=384, name='conv3', kernel_size=3, strides=[1,1,1,1])
        conv4 = self.conv_layer(conv3, in_channels=384, out_channels=384, name='conv4', kernel_size=3, strides=[1,1,1,1])
        conv5 = self.conv_layer(conv4, in_channels=384, out_channels=256, name='conv5', kernel_size=3, strides=[1,1,1,1])
        pool5 = self.max_pool(conv5, 'pool1', kernel_size=3,strides=[1,2,2,1])

        fc6 = self.fc_layer(pool5, 25088, 4096, "fc6")  # 25088 = ((224 / (2 ** 5)) ** 2) * 512
        fc6 = tf.nn.dropout(fc6, 0.5)

        fc7 = self.fc_layer(fc6, 4096, 4096, "fc7")
        fc7 = tf.nn.dropout(fc7, 0.5)

        fc8 = self.fc_layer(fc7, 4096, 1000, "fc8")

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


class tf_cam(tf_model):
    def __init__(self,N_classes):
        self.N_classes = 3
    def get_training_model(self):
        height = 28
        width = 28
        x = tf.placeholder(tf.float32, [None, height, width, 3])
        conv1_1 = self.conv_layer(x, 3, 64, "conv1_1")
        pool1 = self.max_pool(conv1_1, 'pool1')
        conv2_1 = self.conv_layer(pool1, 64, 32, "conv2_1")
        pool2 = self.max_pool(conv2_1, 'pool1')

        conv_shape = 7 * 7 * 32
        conv_layer_flat = tf.reshape(pool2, [-1, conv_shape])
        fc3 = self.fc_layer(conv_layer_flat, conv_shape, 32, "fc6")  # 25088 = ((224 / (2 ** 5)) ** 2) * 512
        # y = fc3
        fc4 = self.fc_layer(fc3,32,3,'fc3')
        y = fc4
        # y = pool2
        # y = fc3
        # params = [
        #     W1_1,b1_1,
        #     W2_1,b2_1,
        #     W3_1,b3_1,
        #     W4_1,b4_1,
        #     ]
        # layers = [2]

        params = {}
        for var in tf.all_variables():
            params[var.name] = var
        return x, y,params

    def get_loss(self,y,y_):
        loss = tf.nn.softmax_cross_entropy_with_logits(y,y_)
        return loss

    def get_accuracy(self,y,y_):
        best = tf.argmax(y,1)
        correct = tf.argmax(y_,1)


class tf_fcn(tf_model):
    def __init__(self,N_classes,height,width):
        self.N_classes = N_classes
        if(N_classes==21):
            self.fc_size = 256
        else:
            self.fc_size = 4096
        self._layers = {}



    def get_training_model(self):
        layers = {}
        x = tf.placeholder(tf.float32, [None, 224, 224, 3])

        #conv: 3 => 64, kernel 3, pad 1
        layers['conv1_1'] = self.conv_layer(x, 3, 64, "conv1_1",kernel_size=3,trainable=False)
        #conv: 64 => 64, kernel 3, pad 1
        layers['conv1_2'] = self.conv_layer(layers['conv1_1'], 64, 64, "conv1_2",kernel_size=3,trainable=False)
        #max pool: kernel 2, stride 2
        layers['pool1'] = self.max_pool(layers['conv1_2'], 'pool1')


        #112x112
        #conv: 64 => 128, kernel 3, pad 1
        layers['conv2_1'] = self.conv_layer(layers['pool1'], 64, 128, "conv2_1",kernel_size=3,trainable=False)
        #conv: 128 => 128, kernel 3, pad 1
        layers['conv2_2'] = self.conv_layer(layers['conv2_1'], 128, 128, "conv2_2",kernel_size=3,trainable=False)
        #max pool: kernel 2, stride 2
        layers['pool2'] = self.max_pool(layers['conv2_2'], 'pool2')


        #56x56
        #conv: 128 => 256, kernel 3, pad 1
        layers['conv3_1'] = self.conv_layer(layers['pool2'], 128, 256, "conv3_1",kernel_size=3,trainable=False)
        #conv: 256 => 256, kernel 3, pad 1
        layers['conv3_2'] = self.conv_layer(layers['conv3_1'], 256, 256, "conv3_2",kernel_size=3,trainable=False)
        #conv: 256 => 256, kernel 3, pad 1
        layers['conv3_3'] = self.conv_layer(layers['conv3_2'], 256, 256, "conv3_3",kernel_size=3,trainable=False)
        #max pool: kernel 2, stride 2
        layers['pool3'] = self.max_pool(layers['conv3_3'], 'pool3')


        #28x28
        #conv: 256 => 512, kernel 3, pad 1
        layers['conv4_1'] = self.conv_layer(layers['pool3'], 256, 512, "conv4_1",kernel_size=3,trainable=False)
        #conv: 512 => 512, kernel 3, pad 1
        layers['conv4_2'] = self.conv_layer(layers['conv4_1'], 512, 512, "conv4_2",kernel_size=3,trainable=False)
        #conv: 512 => 512, kernel 3, pad 1
        layers['conv4_3'] = self.conv_layer(layers['conv4_2'], 512, 512, "conv4_3",kernel_size=3,trainable=False)
        #max pool: kernel 2, stride 2
        layers['pool4'] = self.max_pool(layers['conv4_3'], 'pool4')

        #14x14
        #conv: 512 => 512, kernel 3, pad 1
        layers['conv5_1'] = self.conv_layer(layers['pool4'], 512, 512, "conv5_1",kernel_size=3,trainable=False)
        #conv: 512 => 512, kernel 3, pad 1
        layers['conv5_2'] = self.conv_layer(layers['conv5_1'], 512, 512, "conv5_2",kernel_size=3,trainable=False)
        #conv: 512 => 512, kernel 3, pad 1
        layers['conv5_3'] = self.conv_layer(layers['conv5_2'], 512, 512, "conv5_3",kernel_size=3,trainable=False)
        #max pool: kernel 2, stride 2
        layers['pool5'] = self.max_pool(layers['conv5_3'], 'pool5')

        #7x7
        layers['conv6'] = self.conv_layer(layers['pool5'],512, 4096, "conv6",kernel_size=7,padding='SAME',trainable=True)


        #1x1
        layers['conv7'] = self.conv_layer(layers['conv6'],4096, 4096, "conv7",kernel_size=1,padding='SAME',trainable=True)

        #1x1
        layers['conv8'] = self.conv_layer(layers['conv7'],4096, self.N_classes, "conv8",kernel_size=1,padding='SAME',trainable=True)

        out_features = layers['pool4'].get_shape()[3].value #512
        print(layers['pool4'].get_shape())
        layers['deconv1'] = self.deconv_layer(layers['conv8'],out_features,tf.shape(layers['pool4']),'deconv1',kernel_size=4,trainable=True)
        layers['deconv1'] = tf.add(layers['deconv1'],layers['pool4'])

        out_features = layers['pool3'].get_shape()[3].value #256
        layers['deconv2'] = self.deconv_layer(layers['deconv1'],out_features,tf.shape(layers['pool3']),'deconv2',kernel_size=4,trainable=True)
        layers['deconv2'] = tf.add(layers['deconv2'],layers['pool3'])

        out_features = self.N_classes
        input_shape = tf.shape(x)
        output_shape = tf.pack([input_shape[0],input_shape[1],input_shape[2],out_features])
        layers['deconv3'] = self.deconv_layer(layers['deconv2'],out_features,output_shape,'deconv3',kernel_size=16,strides=[1,8,8,1],trainable=True)

        layers['pred'] = tf.argmax(layers['deconv3'], dimension=3, name="deconv3")
        y = layers['deconv3']

        # layers['deconv3'] = tf.add(layers['deconv2'],layers['pool3'])



        #fc6 decon
        # layers['fc6-deconv'] = self.deconv_layer(layers['fc7'],tf.shape(layers['pool5']),512,'fc6-deconv',kernel_size=7,strides=[1,2,2,1],trainable=True)




        # layers['conv6'] = self.conv_layer(layers['pool5'],512, 4096, "conv6_1",kernel_size=7,padding='SAME',trainable=False)
        # layers['conv6'] = tf.nn.dropout(layers['conv6'], 0.5)

        # layers['conv7'] = self.conv_layer(layers['conv6'],4096, 4096, "conv7_1",kernel_size=7,padding='SAME',trainable=False)
        # layers['conv7'] = tf.nn.dropout(layers['conv7'], 0.5)

        # layers['score_fr'] = self.conv_layer(layers['conv7'],4096, self.N_classes, "score_fr",kernel_size=1,padding='SAME',trainable=True)
        # layers['score_fr'] = tf.nn.dropout(layers['score_fr'], 0.5)

        # layers['upscore2'] = self.deconv_layer(layers['score_fr'],tf.shape(layers['pool4']),self.N_classes,'deconv',kernel_size=4,strides=[1,2,2,1],trainable=True)


        # layers['score_pool4'] = self.conv_layer(layers['pool4'],512, self.N_classes, "score_pool4",kernel_size=1,padding='SAME',trainable=True)

        # layers['fuse_pool4'] = tf.add(layers['upscore2'], layers['score_pool4'])

        # layers['fin'] = self.deconv_layer(layers['fuse_pool4'],tf.shape(x),self.N_classes,'upscore32',kernel_size=32,strides=[1,16,16,1],trainable=True)

        # layers['pred2'] = tf.argmax(layers['fin'],dimension=3)

        #7x7
        #deconv5_1
        # layers['deconv5_1'] = self.deconv_layer(layers['fc6-deconv'],tf.shape(layers['pool5']),512,'fc6-deconv',kernel_size=3,strides=[1,2,2,1],trainable=True)



        # y=layers['deconv5_1']

        # layers['conv7'] = self.conv_layer(layers['conv6'],4096, 4096, "conv7_1",kernel_size=7,padding='SAME',trainable=False)
        # layers['conv7'] = tf.nn.dropout(layers['conv7'], 0.5)



        # self.deconv_layer(layers['fc6'], 3,28,28,self.N_classes, 'deconv7', ksize=3, stride=[1,2,2,1])

        # y = layers['fin']


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

        # cross_entropy = -y_ * tf.log(tf.nn.softmax(y))
        # # cross_entropy = [1.,2.]
        # return tf.reduce_mean(cross_entropy)
        return tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(y,
                                                                          tf.squeeze(y_, squeeze_dims=[3]),
                                                                          name="entropy")))

    # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    # if FLAGS.debug:
    #     # print(len(var_list))
    #     for grad, var in grads:
    #         utils.add_gradient_summary(grad, var)
    def get_accuracy(self,y,y_):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        A = tf.reduce_sum(tf.cast(correct_prediction, "float"))
        return A

    def get_optimizer(self,loss,lr):
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=lr,
            initial_accumulator_value=0.1,
            use_locking=False,
            name='Adagrad')
        return optimizer.minimize(loss)

class tf_cam2(tf_model):
    def __init__(self,N_classes,height=28,width=28):
        self.N_classes = N_classes
        self.height = height
        self.width = width
        self._layers = {}
        self._params = {}
    def get_training_model(self):
        x = tf.placeholder(tf.float32, [None, self.height, self.width, 3])
        layers = {}
        layers['conv1'] = self.conv_layer(x, 3, 64, "conv1")
        layers['pool2'] = self.max_pool(layers['conv1'], 'pool2')
        layers['conv3'] = self.conv_layer(layers['pool2'], 64, 32, "conv3")
        layers['pool4'] = self.max_pool(layers['conv3'], 'pool4')

        conv_shape = 7 * 7 * 32
        layers['conv_layer_flat'] = tf.reshape(layers['pool4'], [-1, conv_shape])
        layers['fc5'] = self.fc_layer(layers['conv_layer_flat'], conv_shape, 32, "fc5")  # 25088 = ((224 / (2 ** 5)) ** 2) * 512
        # y = fc3
        layers['fc6'] = self.fc_layer(layers['fc5'],32,self.N_classes,'fc6')


        layers['deconv7'] = self.deconv_layer(layers['fc6'], 3,28,28,self.N_classes, 'deconv7', ksize=3, stride=[1,2,2,1])

        # y = layers['fc6']
        y = layers['deconv7']

        self._layers = layers

        for var in tf.all_variables():
            self._params[var.name] = var
        return x, y,self._params

    @property
    def layers(self):
        return self._layers

    def get_loss(self,y,y_):
        loss = tf.nn.softmax_cross_entropy_with_logits(y,y_)
        return tf.reduce_mean(loss)

    def get_accuracy(self,y,y_):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        return tf.reduce_sum(tf.cast(correct_prediction, "float"))

    def get_optimizer(self,loss,lr):
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=lr,
            initial_accumulator_value=0.1,
            use_locking=False,
            name='Adagrad')
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=lr,
            rho=0.95,
            epsilon=1e-05,
            use_locking=False,
            name='Adadelta')
        return optimizer.minimize(loss)


class tf_blm(tf_model):
    def __init__(self,N_classes):
        self.N_classes = N_classes
        self.bl_model = []


    def max_pool(self,bottom, name, kernel_size=2,strides=[1,2,2,1]):
        input_layer = self.bl_model[-1]
        dico = {}
        dico['name'] = name
        dico['type'] = 'max_pool'
        dico['weights_name'] = None
        dico['biais_name'] = None
        dico['dim_x'] = input_layer['dim_x'] / kernel_size
        dico['dim_y'] = input_layer['dim_y'] / kernel_size
        dico['dim_z'] = input_layer['dim_z']
        dico['kernel_size_x'] = kernel_size
        dico['kernel_size_y'] = kernel_size
        return super().max_pool(bottom, name, kernel_size,strides)
    def fc_layer(self,bottom, in_size, out_size, name):
        input_layer = self.bl_model[-1]
        dico = {}
        dico['name'] = name
        dico['type'] = 'fully_connected'
        dico['weights_name'] = name+'_weight'
        dico['biais_name'] = name+'_biais'
        dico['dim_x'] = input_layer['dim_x']
        dico['dim_y'] = None
        dico['dim_z'] = None
        return super().fc_layer(bottom, in_size, out_size, name)

    def conv_layer(self,bottom, in_channels, out_channels, name, kernel_size=3, strides=[1,1,1,1],padding='SAME'):
        input_layer = self.bl_model[-1]
        dico = {}
        dico['name'] = name
        dico['type'] = 'convolution2d'
        dico['weights_name'] = name+'_weight'
        dico['biais_name'] = name+'_biais'
        if(padding=='VALID'):
            dico['dim_x'] = input_layer['dim_x'] - kernel_size
            dico['dim_y'] = input_layer['dim_y'] - kernel_size
        else:
            dico['dim_x'] = input_layer['dim_x']
            dico['dim_y'] = input_layer['dim_y']
        dico['dim_z'] = out_channels
        dico['kernel_size_x'] = kernel_size
        dico['kernel_size_y'] = kernel_size
        dico['activation'] = 'relu'
        self.bl_model.append(dico)
        return super().conv_layer(bottom, in_channels, out_channels, name, kernel_size,strides,padding)
    def get_training_model(self):
        height = 28
        width = 28
        layers = {}
        N_input_chanels = 1
        x = tf.placeholder(tf.float32, [None, height, width, N_input_chanels])

        dico = {}
        dico['name'] = 'input'
        dico['dim_x'] = height
        dico['dim_y'] = width
        dico['dim_channels'] = N_input_chanels
        self.bl_model.append(dico)
        layers['conv1'] = self.conv_layer(x, N_input_chanels, 6, "conv1",5,padding='VALID')
        layers['pool1'] = self.max_pool(layers['conv1'], 'pool1')
        layers['pad1'] = tf.pad(layers['pool1'],[[0,0],[1,1],[1,1],[0,0]])

        layers['conv2'] = self.conv_layer(layers['pad1'], 6, 12, "conv2",5,padding='VALID')
        layers['pool2'] = self.max_pool(layers['conv2'], 'pool1')

        conv_shape = 5 * 5 * 12
        conv_layer_flat = tf.reshape(layers['pool2'], [-1, conv_shape])
        layers['fc3'] = self.fc_layer(conv_layer_flat, conv_shape, 10, "fc3")  # 25088 = ((224 / (2 ** 5)) ** 2) * 512
        # y = fc3
        # layers['fc4'] = self.fc_layer(layers['fc3'],32,3,'fc3')
        y = layers['fc3']
        print(self.bl_model)
        # y = pool2
        # y = fc3
        # params = [
        #     W1_1,b1_1,
        #     W2_1,b2_1,
        #     W3_1,b3_1,
        #     W4_1,b4_1,
        #     ]
        # layers = [2]
        self._layers = layers
        params = {}
        for var in tf.all_variables():
            params[var.name] = var
        return x, y,params

    @property
    def layers(self):
        return self._layers

    def get_loss(self,y,y_):
        loss = tf.nn.softmax_cross_entropy_with_logits(y,y_)
        return loss

    def get_accuracy(self,y,y_):
        best = tf.argmax(y,1)
        correct = tf.argmax(y_,1)


import os
import numpy as np
import logging
from math import ceil
class tf_copy:
    def __init__(self,N_classes,height,width):

        self.N_classes = N_classes
        path = 'weights'
        # print path

        # print path
        path = os.path.join(path, "vgg19.npy")
        vgg16_npy_path = path
        logging.info("Load npy file from '%s'.", vgg16_npy_path)

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.wd = 5e-4
        print("npy file loaded")

    def get_training_model(self):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        x =tf.placeholder(tf.float32, [None, 224, 224, 3])
        debug=False
        train=True
        random_init_fc8=True
        num_classes = self.N_classes
        self.conv1_1 = self._conv_layer(x, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")

        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)

        print('fc6')
        self.fc6 = self._fc_layer(self.pool5, "fc6")

        if train:
            self.fc6 = tf.nn.dropout(self.fc6, 0.5)

        self.fc7 = self._fc_layer(self.fc6, "fc7")
        if train:
            self.fc7 = tf.nn.dropout(self.fc7, 0.5)

        if random_init_fc8:
            self.score_fr = self._score_layer(self.fc7, "score_fr",
                                              num_classes)
        else:
            self.score_fr = self._fc_layer(self.fc7, "score_fr",
                                           num_classes=num_classes,
                                           relu=False)

        self.pred = tf.argmax(self.score_fr, dimension=3)

        self.upscore2 = self._upscore_layer(self.score_fr,
                                            shape=tf.shape(self.pool4),
                                            num_classes=num_classes,
                                            debug=debug, name='upscore2',
                                            ksize=4, stride=2)

        self.score_pool4 = self._score_layer(self.pool4, "score_pool4",
                                             num_classes=num_classes)

        self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

        self.upscore32 = self._upscore_layer(self.fuse_pool4,
                                             shape=tf.shape(x),
                                             num_classes=num_classes,
                                             debug=debug, name='upscore32',
                                             ksize=32, stride=16)

        self.pred_up = tf.argmax(self.upscore32, dimension=3)

    def _max_pool(self, bottom, name, debug):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            _activation_summary(relu)
            return relu

    def _fc_layer(self, bottom, name, num_classes=None,
                  relu=True, debug=False):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                                  num_classes=num_classes)
            else:
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, num_classes=num_classes)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)
            _activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)

            print(name,bottom.get_shape(),bias.get_shape(),filt.get_shape())
            return bias

    def _score_layer(self, bottom, name, num_classes):
        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            if name == "score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "score_pool4":
                stddev = 0.001
            # Apply convolution
            w_decay = self.wd
            weights = self._variable_with_weight_decay(shape, stddev, w_decay)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self._bias_variable([num_classes], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            _activation_summary(bias)
            print(name,bottom.get_shape(),bias.get_shape(),weights.get_shape())
            return bias

    def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.pack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)
            print(name,bottom.get_shape(),deconv.get_shape())
        _activation_summary(deconv)
        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        print(weights.shape)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        print('Layer name: %s' % name)
        print('Layer shape: %s' % str(shape))
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
                                  name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        if name == 'fc8':
            bias_wights = self._bias_reshape(bias_wights, shape[0],
                                             num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        return tf.get_variable(name="biases", initializer=init, shape=shape)

    def get_fc_weight(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.mul(tf.nn.l2_loss(var), self.wd,
                                  name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`
        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.
        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.
        Consider reordering fweight, to perserve semantic meaning of the
        weights.
        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes
        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _variable_with_weight_decay(self, shape, stddev, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.
        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.
        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        return tf.get_variable(name='biases', shape=shape,
                               initializer=initializer)

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = self._summary_reshape(weights, shape,
                                            num_new=num_classes)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="weights", initializer=init, shape=shape)


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
