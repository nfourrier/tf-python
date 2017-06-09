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

    def __str__(self):
        return "Layers {0}\nVariables {1}".format(self.layers,self.variables)

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

    def set_input_tensor(self,tensor):
        self.inp = tf.identity(tensor,name='input')
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
        self.out = tf.identity(tf_layer.out,name='output')
        self._layers['output'] = self.out
        self._layers_list.append([self.out.name,self.out,'output'])


        
        for var in tf.global_variables(): ### old version was tf.all_variables()
            
            self._variables[var.name] = var

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
            session.run(self._variables[name].assign(f[gp][dat].value))
            
        f.close()
        print('\t ... weights loaded')


