import tensorflow as tf


import dasakl.nn.layer as lay
class mmodel(object):
    def __init__(self):
        '''
            Build tensorflow model given a list of layers with parameters or a cfg file.
        '''
        self.name = 'default'
        self._layers = {}
        self._layers_list = []
        self._variables = {}

    def parse(self,cfg_file):
        ''' 
            Parse cfg file to extract the layer type and parameters
        '''
        from dasakl.nn.parser import get_layers
        layers = get_layers(cfg_file)
        return layers

    def get_input(self,dim0,dim1,dim2):
        '''
            Returns the tensor containing the input.
            Note that the current configuration assumes a 2d rgb images
            The input is a 4 dimensional tensor:
                first dimension (None): corresponds to the batch
                second dimension (dim0): corresponds to the width of the image
                third dimension (dim1): corresponds to the height of the image
                fourth dimension (dim2): corresponds to the channels of the image
        '''
        self.inp = tf.placeholder(tf.float32,[None, dim0, dim1, dim2])
        return self.inp


    def set_input(self,dim):
        '''
            Defines the input tensor given a list of dimension (dim)
        '''
        self.inp = tf.placeholder(tf.float32,[None]+dim,name='input')
        return self.inp


    def get_model(self,layers):
        '''
            Generates and returns the model
            Before using this function the input tensor must exist (self.inp)
            For each layer (given in the list layers), build the layer according to its parameter
            It constructs three objects:
                - layers (dictionary): keys are defined by index_typeLayer. The dictionary contains the output tensor
                - layers_list (list): contains the output tensors for each layer (easy to manipulate when the network is linear)
                - variables (dictionary): keys are the variable name. The dictionary contains the variable tensor.
        '''

        inp = [self.inp]



        N_layer = 0
        # print(inp[0].get_shape(),'input')
        self._layers_list.append([self.inp.name,self.inp,'input'])
        self._layers['input'] = self.inp
        for layer in layers:
            param = layer

            if(N_layer==0):
                param[3] = [self.inp]
            else:
                # print(param[-1])
                param[3] = [self._layers['{}_{}'.format(param[3][idx][0],param[3][idx][1])] for idx in range(len(param[3]))]
            
            tf_layer = lay.create_layer(*param)
            self._layers['{}_{}'.format(param[0],param[1])] = tf_layer.out
            self._layers_list.append([tf_layer.out.name,tf_layer.out,'{}_{}'.format(param[0],param[1])])
            
            N_layer += 1


            # if(param[0]=='fullyconnected'):
            #     self._layers['flatten_input'] = tf_layer.flatten[0]
            #     self._layers['flatten_output'] = tf_layer.flatten[1]
                # self._layers['flatten_output_transpose'] = tf_layer.flatten[1]
        self.out = tf.identity(tf_layer.out,name='output')
        self._layers['output'] = self.out
        self._layers_list.append([self.out.name,self.out,'output'])


        
        for var in tf.global_variables(): ### old version was tf.all_variables()
            
            self._variables[var.name] = var

        # return self.inp,tf_layer.out,self._variables
        return self.inp,self.out,self._variables

    @property
    def layers(self):
        '''
            Returns the model layers as a list
        '''
        return self._layers_list

    @property
    def layers_dico(self):
        '''
            Returns the model layers as a dictionary (keys are the layer name following this nomenclature: index_layerType)
        '''
        return self._layers

    @property
    def variables(self):
        '''
            Returns the model variabls as a dictionary.
        '''
        return self._variables

    def load_weights(self,session,filename,layers=None):
        '''
            Given a session and a filename (hdf5 format), load the weights into the model  variables
        ''' 
        import h5py
        f = h5py.File(filename,'r')
        print('Load weights from {} \n \t last modified: {} \n \t ...'.format(filename,time.ctime(os.path.getmtime(filename))))
        if(isinstance(layers,type(None))):
            layers = self._variables.keys() 
        

        for name in layers:
            gp,dat = name.split("/")
           
            # print(name,gp,dat,var_dict[name].get_shape(),f[gp][dat].shape)
            # print(name,f[gp][dat].value.mean())
            # print(name)
            # print(f[gp][dat].value.mean())
            # print(f[gp][dat].value)
            # print(name,np.prod(f[gp][dat].value.shape),self._variables[name].get_shape(),self._variables[name],f[gp][dat].value.dtype)

            session.run(self._variables[name].assign(f[gp][dat].value))
            # print(f[gp][dat].value.mean())
            
        f.close()
        print('\t ... weights loaded')
        # exit()


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


