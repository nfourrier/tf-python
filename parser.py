import numpy as np
import pickle
import os
from copy import deepcopy
from array import array

FORM = '{:>5} | {:<18} | {:<20} | {}'
FORM_ = '{}+{}+{}+{}'
LINE = FORM_.format('-'*6, '-'*20, '-'*22, '-'*15) 
HEADER = FORM.format(
    'Index', 'Layer description', 'Output size','Input')
EXT = '.txt'

def _parser(model):
    """
    Read the .cfg file to extract layers into `layers`
    as well as model-specific parameters into `meta`
    """
    def _parse(l, i = 1):
        return l.split('=')[i].strip()

    with open(model, 'rb') as f:
        lines = f.readlines()

    lines = [line.decode() for line in lines]   
    
    meta = dict(); layers = list() # will contains layers' info
    h, w, c = [int()] * 3; layer = dict()
    for line in lines:
        line = line.strip()
        line = line.split('#')[0]
        if '[' in line:
            if layer != dict(): 
                if layer['type'] == '[input]':
                    layer_keys = layer.keys()
                    input_list = []
                    for inp in layers_list:
                        tmp = inp[1].split(',')

                        is_numeric = [x.isnumeric() for x in tmp]
                        inp_dim = []
                        N_non_numeric = 0
                        inp_type = 'float'
                        inp_name = inp[0]
                        for idx in range(len(tmp)):
                            x = tmp[idx]
                            if(is_numeric[idx]):
                                inp_dim.append(int(x))
                            else:
                                N_non_numeric = N_non_numeric + 1
                                inp_type = x.lower()

                        if(N_non_numeric>1):
                            print("Error in cfg file with:\n\t{}".format(line))
                            print("Input contains at most one string")
                            exit()
                        input_list.append([inp_name,inp_type,inp_dim])
                else:
                    layers += [layer]            
            layer = {'type': line}
            layers_list = []
        else:

            try:
                key = _parse(line, 0)
                val = _parse(line, 1)
                layer[key] = val
                layers_list.append([key,val])
            except:
                'no'

    meta.update(layer) # last layer contains meta info
    meta['inp_size'] = input_list[0][2]
    
    return layers, meta, input_list


def get_layers(model):
    """
    return a list of `layers` objects 
    """
    args = [model]
    cfg_layers = _cfg_yielder(*args)
    meta = dict(); layers = list()
    NEW_LINE = '\t{}\n'
    msg = '\n'
    msg += NEW_LINE.format('Model summary')
    msg += NEW_LINE.format('=============')
    msg += NEW_LINE.format(HEADER)
    msg += NEW_LINE.format(LINE)
    for i, info in enumerate(cfg_layers):       
        ### Recieved meta
        if i == 0: 
            meta = info
            # tmp = FORM.format('','input',' x '.join(['{:>4}'.format(x) for x in meta['inp_size']]),'')    
            # msg += NEW_LINE.format(tmp)
            # msg += NEW_LINE.format(LINE)
            continue
        ### Recieved input_list
        if i == 1:
            input_list = info
            for inp in input_list:
                tmp = FORM.format('',inp[0],' x '.join(['{:>4}'.format(x) for x in inp[2]]),'')    
                msg += NEW_LINE.format(tmp)
            msg += NEW_LINE.format(LINE)
            continue

        tmp = FORM.format(info[1],info[0],' x '.join(['{:>4}'.format(x) for x in info[2]]),', '.join(['{}'.format(x[1]) for x in info[3]]))
        # else: new = create_darkop(*info)
        msg += NEW_LINE.format(tmp)
        msg += NEW_LINE.format(LINE)
        layers.append(deepcopy(info))

    return meta, input_list, layers, msg    



def _cfg_yielder(model):
    """
    yielding each layer information to initialize `layer`
    """

    def _initialization(d):
        initialization = d.get("initialization","truncated_normal,0.2")
        initialization = initialization.split(",")
        if(len(initialization))>1:
            initialization = [initialization[0]]+[float(x) for x in initialization[1:]]        
        return initialization

    def _activation(d):
        activation = d.get('activation','identity,-1,-1')
        activation = activation.split(",")
        if(len(activation))>1:
            activation = [activation[0]]+[float(x) for x in activation[1:]]    
        return activation       

    def _trainable(d):
        trainable = (d.get('trainable', 'True').lower()=='true')
        return trainable

    layers, meta, input_list = _parser(model)
    yield meta
    yield input_list
    dim = [-1,-1,-1]
    output_layer = {}

    # Start yielding
    flat = False # flag for 1st dense layer
    conv = '.conv.' in model
    layer_type = ''

    layer_type_key = 'layer_type'

    #### INITIALIZE OUTPUT_LAYERS WITH INPUTS
    prefix_read = ''
    prefix_idx = -1
    for inp in input_list:
        tag = inp[4]
        dim = inp[2]

        output_layer[tag] = [inp[0],-1,deepcopy(dim),prefix_read,prefix_idx]


    dim = [-1,-1,-1]
    h, w, c = dim

    idx_layers = {}
    idx_layers['layer'] = 0
    idx_layers['preprocess'] = 0
    idx_layers['custom'] = 0
    idx_layers['loss'] = 0
    idx_layers['optimizer'] = 0

    for i, d in enumerate(layers):
        output_layer_keys = output_layer.keys()
        routes_read = d.get('input_tags',-1)
        if(routes_read==-1):
            routes_read = d.get('layers',-1)



        routes = []
        if type(routes_read) is str:

            routes_read = [x.strip() for x in routes_read.split(',')]
            for x in routes_read:
                if(x in output_layer_keys):
                    routes.append(output_layer[x])
                else:
                    print('Error in routing for layer: {} is not a know tag.\n{}'.format(x,'\n'.join(['\t{}:{}'.format(k,v) for k,v in d.items()])))
                    exit()
        elif(routes_read>-1): 
            print("try to use string for tags")
            routes = [output_layer[routes_read]]
        else:
            routes = [[layer_type,i-1,deepcopy(dim),prefix_read,prefix_idx]]


        summary_read = d.get("summary",False)
        



        outp_layer = d.get('layer_output','last_layer')
        if(outp_layer=='last_layer'):
            outp_layer = d.get('tags','last_layer')

        if(outp_layer!='last_layer'): output_layer[outp_layer] = [layer_type,i,deepcopy(dim),prefix_read,prefix_idx]
        # print(layer_type,dim)
        # if(outp_layer>-1): print(output_layer)
        # d['_size'] = list([h, w, c, l, flat])
    if not flat: meta['out_size'] = [h, w, c]
    else: meta['out_size'] = l
