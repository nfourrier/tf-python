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

    def set_input(self,dim):
        '''
            Defines the input tensor given a list of dimension (dim)
        '''
        dim = [x for x in dim if x != 0]
        dim = [x if x > -1 else None for x in dim]
        
        self.inp = tf.placeholder(tf.float32,[None]+dim,name='input')
        return self.inp


    def set_input_tensor(self,tensor):
        self.inp = tf.identity(tensor,name='input')
        return self.inp

    def set_preprocess(self,fct):
        self.preprocess = fct
    def set_postprocess(self,fct):
        self.postprocess = fct

    def get_model(self,layers,meta):
        '''
            Generates and returns the model
            Before using this function the input tensor must exist (self.inp)
            For each layer (given in the list layers), build the layer according to its parameter
            It constructs three objects:
                - layers (dictionary): keys are defined by index_typeLayer. The dictionary contains the output tensor
                - layers_list (list): contains the output tensors for each layer (easy to manipulate when the network is linear)
                - variables (dictionary): keys are the variable name. The dictionary contains the variable tensor.
        '''        
        inp = self.inp



        N_layer = 0

        inp_layer_list = []
        inp_layers = []
        for inp_idx in self.inp:
            name_idx = inp_idx.name
            inp_layer_list.append([name_idx,inp_idx,name_idx])
            inp_layers.append(inp_idx)
        self._layers_list.append(inp_layer_list)
        self._layers['input'] = inp_layers
        
        if(not isinstance(self.preprocess,type(None))):
            tmp = self.preprocess(self.inp,meta)
        else:
            tmp = self.inp

        if(not isinstance(tmp,type(['list']))):
            tmp = [tmp]
        
        inp_pp_layer_list = []
        self.inp_preprocess = []
        for inp_idx in tmp:
            name_idx = inp_idx.name
            inp_pp_layer_list.append([name_idx,inp_idx,name_idx])
            self.inp_preprocess.append(inp_idx)
            self._layers_list.append(inp_pp_layer_list)
        self._layers['input_preprocess'] = self.inp_preprocess


        
        for layer in layers:
            
            param = layer
            
            if(N_layer==0):
                param[3] = self.inp_preprocess
            else:
                param[3] = [self._layers['{}_{}'.format(param[3][idx][0],param[3][idx][1])] for idx in range(len(param[3]))]
            tf_layer = lay.create_layer(*param)
            self._layers['{}_{}'.format(param[0],param[1])] = tf_layer.out
            self._layers_list.append([tf_layer.out.name,tf_layer.out,'{}_{}'.format(param[0],param[1])])
            
            
            N_layer += 1



        self.out = tf.identity(tf_layer.out,name='output')
        self._layers['output'] = self.out
        self._layers_list.append([self.out.name,self.out,'output'])

        if(not isinstance(self.postprocess,type(None))):
            tmp = self.postprocess(self.inp,self.out,meta)
        else:
            tmp = self.out
        self.out_postprocess = tf.identity(tmp,name='output_postprocess')
        self._layers['output_postprocess'] = self.out_postprocess
        self._layers_list.append([self.out_postprocess.name,self.out_postprocess,'output_postprocess'])

        for var in tf.global_variables(): ### old version was tf.all_variables()
            self._variables[var.name] = var

    
        return self.inp,self.inp_preprocess,self.out,self.out_postprocess,self._variables

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
            # print(self._variables[name],type(self._variables[name]))

            session.run(self._variables[name].assign(f[gp][dat].value))
            
        f.close()
        print('\t ... weights loaded')
