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
        
        for var in tf.global_variables(): ### old version was tf.all_variables()
            # print(var.name)
            self._variables[var.name] = var

        return self.inp,self._layers['{}_{}'.format(param[0],param[1])],self._variables


    def load_weights(self,session,filename,layers=None):
        import h5py
        f = h5py.File(filename,'r')
        print('Load weights from {} \n \t last modified: {} \n \t ...'.format(filename,time.ctime(os.path.getmtime(filename))))
        if(isinstance(layers,type(None))):
            layers = self._variables.keys() 
        for name in layers:
            gp,dat = name.split("/")
            print(name,np.prod(f[gp][dat].value.shape),self._variables[name].get_shape())
            session.run(self._variables[name].assign(f[gp][dat].value))

        f.close()
        print('\t ... weights loaded')
        


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


