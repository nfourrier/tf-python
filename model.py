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

    def build_inputs(self,input_list):
        '''
        Build placeholder defining the inputs of the model. 
        Main input is assumed to be the first element of the list
        Inputs can be the image feed but also additional parameters such as hyperparamters (learning rate)
        input_list contains a list of inputs of the following form:
            ['name','type',[dim1,dim2,...,dimN]]
        '''
        FORM = '{:>5} | {:<18} | {:<28} | {}'
        FORM_ = '{}+{}+{}+{}'
        LINE = FORM_.format('-'*6, '-'*20, '-'*30, '-'*20) 
        HEADER = FORM.format(
            'Index', 'Name', 'Dimension','Type')
        NEW_LINE = '\t{}\n'


        msg = '\n'
        msg += NEW_LINE.format('Input summary')
        msg += NEW_LINE.format('=============')
        msg += NEW_LINE.format(HEADER)
        msg += NEW_LINE.format(LINE)


        self.inp = []
        idx = 0
        for x in input_list:
            x_name = x[0]
            x_type = x[1].lower()
            x_dim = x[2]
            x_dim = [y for y in x_dim if y != 0]
            x_dim = [y if y > -1 else None for y in x_dim]
            x_dim = [None]+x_dim
            x_summary = x[3]
            if(isinstance(x_summary,type('string'))):
                if("false"==x_summary.lower()):
                    x_summary = False
                else: 
                    x_summary = True
            elif(isinstance(x_summary,type(int(3)))):
                if(x_summary>0):
                    x_summary = True
                else:
                    x_summary = False

            if("float" in x_type):
                if("float64"==x_type):
                    tf_type = tf.float64
                else:
                    tf_type = tf.float32
            elif("string" in x_type):
                tf_type = tf.string
            elif("int" in x_type):
                if("int64"==x_type):
                    tf_type = tf.int64
                elif("uint8"==x_type):
                    tf_type = tf.uint8
                else:
                    tf_type = tf.int32
            else:
                print("INPUT BUILD ERROR\n\t{}\n\tPlease specify a different type. {} is not in list [float,string,int]".format(x,x_type))
                exit()
            # self.inp.append(tf.placeholder(tf_type,[None]+x_dim,name='input_{}'.format(idx)))
            tf_x = tf.placeholder(tf_type,x_dim,name=x_name)
            if(x_summary):
                summary = tf.summary.tensor_summary("Input", tf_x, max_outputs=1)
            self.inp.append(tf_x)
            # self._layers_list.append([tf_x.name,tf_x,x_name])
            # self._layers[x_name] = tf_x
            self._inputs[x_name] = tf_x
            self._inputs_list.append([tf_x.name,tf_x,x_name])
            self._tensors[x_name] = tf_x
            self._tensors_list.append([tf_x.name,tf_x,x_name])            
            tmp = FORM.format(idx,tf_x.name,' x '.join(['{:>4}'.format(x) if(isinstance(x,type(1))) else '{}'.format(x) for x in tf_x.get_shape().as_list()]),tf_x.dtype)
            msg += NEW_LINE.format(tmp)
            idx = idx + 1
        msg += NEW_LINE.format(LINE)
        self.input_text = msg
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
