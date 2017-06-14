import tensorflow as tf
import numpy as np
import cv2
from collections import defaultdict
from copy import deepcopy
import json 
import math 
import pandas as pd
import dasakl
import dasakl.nn.layer as lay
from dasakl.nn.model import mmodel,generic_model,optimizer
from dasakl.nn.parser import get_layers
from dasakl.utils import Timer, DataSet, Log
import os
import h5py



class Detector(object):
    def __init__(self, cfg=None, weights=None, framework=None, training_parameters=None, GPU=True):
        # tf.reset_default_graph()
        if(not isinstance(framework,type(None))):
            if(isinstance(framework,type('string'))):
                framework = fmwk.create_framework(framework)

        ## Fectch cfg file from framework definition
        if(isinstance(cfg,type(None))):
            try: 
                cfg = framework['cfg']
            except Exception as e:
                exit("\t Please specify a cfg file.")

        ## Fetch weights from framework definition
        if(isinstance(weights,type(None))):
            try:
                weights = framework['weights']
            except Exception as e:
                print("\t No weights in the initialization")

        self.cfg = cfg
        self.weights = weights
        self.gpu = GPU
        ## Parse cfg file and initialize the model
        self.name = os.path.basename(cfg).split('.')[0]
        
        self.model = mmodel()
        self.meta,self.layers,self.archi = self.model.parse(cfg)
        # print(self.archi)

    # def initialisation(self):
        self.training = False

        if(not isinstance(framework,type(None))):
            self.load_framework(framework)  

        if(isinstance(training_parameters,type(None))):
            self.model.set_input(self.meta['inp_size'])
        elif(len(training_parameters)>2):
            self.model.set_input(self.meta['inp_size'])
            self.training=True
        else:
            self.advanced_training_init(training_parameters[0],training_parameters[1])
            self.model.set_input_tensor(self.x)
            self.training = True

        ## Load the model from the cfg file
        self.x,self.y,self.var_dict = self.model.get_model(self.layers)
        

        formatted_variables = self.formatted_variables()
        # print(formatted_variables)  
        self.model_text = self.formatted_model()
        print(self.model_text)
        


        # print(tf.global_variables(),len(tf.global_variables()))
        # saver = tf.train.Saver()
        # print(tf.global_variables(),len(tf.global_variables()))
        # exit()
        # for lay in list(self.model.layers.keys()):
        #     print(lay,self.model.layers[lay].shape)
        # exit()
        if(self.gpu):
            tf_config = tf.ConfigProto(
                log_device_placement=False,gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.80,allow_growth = True))
                        )
        else:
            tf_config = tf.ConfigProto(
                log_device_placement=False,device_count = {'GPU': 0})

        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        # file_save = os.path.join(dasakl.DATA_FOLDER,'weights','segnet_init.hdf5')
        # f = h5py.File(file_save,'w')
        # for key in self.var_dict.keys():
        #     A = self.sess.run(self.var_dict[key])
        #     f.create_dataset(key,data=A)
        # f.close()
        # exit()


        if(self.training):
            self.set_training_parameters(training_parameters[0],training_parameters[1])

        ## Load weights if it exists        
        if(not isinstance(weights,type(None))):
            self.model.load_weights(self.sess,weights)

    def load_weights(self,filename,layers=None):
        self.model.load_weights(self.sess,filename,layers)
    def load_session(self,filename=None):

        # lay = 'Adagrad/update_6-fullyconnected/kernel/ApplyAdagrad'
        print(len([n.name  for n in tf.get_default_graph().as_graph_def().node]))
        print([x.name for x in tf.global_variables()])
        A = self.sess.run('3-convolutional/kernel/Adagrad:0',feed_dict={self.x:self.preprocess('/home/gpu/data/shopkins/img80/000040003.png',self.meta)})
        B = self.sess.run('2-convolutional/kernel:0',feed_dict={self.x:self.preprocess('/home/gpu/data/shopkins/img80/000040003.png',self.meta)})
        print(A.shape,A.mean())
        print(B.shape,B.mean())

        import time
        filename = os.path.join(self.session_path,'checkpoint')
        print('Load session from {} \n \t last modified: {} \n \t ...'.format(filename,time.ctime(os.path.getmtime(filename))))
        self.saver.restore(self.sess, 
            tf.train.latest_checkpoint(self.session_path))
        print('\t ... session loaded')
        print(len([n.name  for n in tf.get_default_graph().as_graph_def().node]))
        print([x.name for x in tf.global_variables()])
        A = self.sess.run('3-convolutional/kernel/Adagrad:0',feed_dict={self.x:self.preprocess('/home/gpu/data/shopkins/img80/000040003.png',self.meta)})
        B = self.sess.run('2-convolutional/kernel:0',feed_dict={self.x:self.preprocess('/home/gpu/data/shopkins/img80/000040003.png',self.meta)})
        print(A.shape,A.mean())
        print(B.shape,B.mean())

    def save_graph(self,filename,freeze=True,output_nodes=None):
        def _freeze_graph(model_folder,model_name,output_nodes):
            # We retrieve our checkpoint fullpath
            checkpoint = tf.train.get_checkpoint_state(model_folder,latest_filename='checkpoint')
            input_checkpoint = checkpoint.model_checkpoint_path
            
            # Before exporting our graph, we need to precise what is our output node
            # This is how TF decides what part of the Graph he has to keep and what part it can dump
            # NOTE: this variable is plural, because you can have multiple output nodes
            # output_node_names = "BiasAdd_18" 

            # We clear devices to allow TensorFlow to control on which device it will load operations
            clear_devices = True
            
            # We import the meta graph and retrieve a Saver
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

            # We retrieve the protobuf graph definition
            graph = tf.get_default_graph()
            input_graph_def = graph.as_graph_def()

            # We start a session and restore the graph weights
            with tf.Session() as sess:
                saver.restore(sess, input_checkpoint)

                # We use a built-in TF helper to export variables to constants
                output_graph_def = graph_util.convert_variables_to_constants(
                    sess, # The session is used to retrieve the weights
                    input_graph_def, # The graph_def is used to retrieve the nodes 
                    output_nodes.split(",") # The output node names are used to select the usefull nodes
                ) 

                # Finally we serialize and dump the output graph to the filesystem
                with tf.gfile.GFile(os.path.join(model_folder,model_name+'.pb'), "wb") as f:
                    f.write(output_graph_def.SerializeToString())
                print("%d ops in the final graph." % len(output_graph_def.node))

        path = os.path.abspath(filename)
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)

        # if(freeze):
        #     filename_tmp = 'tmp_'+filename
        # else:
        #     filename_tmp = filename
        basename = filename.split('.')[0]

        saver = tf.train.Saver()
        # self.sess.run(tf.global_variables_initializer())
        
        saver.save(sess=self.sess,save_path=os.path.join(dirname,basename),global_step=0,latest_filename='checkpoint')
        # tf.train.write_graph(self.sess.graph_def, dirname, filename_tmp)

        if(freeze):
            rmFiles = [basename+'-0'+x for x in ['.data-00000-of-00001','.meta','.index']]
            rmFiles.append('checkpoint')
            rmFiles = [os.path.join(dirname,x) for x in rmFiles]
            _freeze_graph(dirname,basename,output_nodes)
            [os.remove(x) for x in rmFiles]
        else:
            tf.train.write_graph(self.sess.graph_def, dirname, filename)
        print('GRAPH SAVED')

    def load_framework(self,framework_dico):
        self.set_preprocess(framework_dico['preprocess'])
        self.set_postprocess(framework_dico['postprocess'])
        self.set_read_output(framework_dico['read_output'])

        self.set_y_true(framework_dico['y_true'])
        self.set_loss(framework_dico['loss'])
        self.set_accuracy(framework_dico['accuracy'])
        
    def close(self):
        self.sess.close()

    def set_preprocess(self,fct):
        self.preprocess = fct

    def set_postprocess(self,fct):
        self.postprocess = fct

    def set_read_output(self,fct):
        self.read_output = fct

    def set_loss(self,fct):
        self.get_loss = fct

    def set_accuracy(self,fct):
        self.get_accuracy = fct

    def set_y_true(self,fct):
        self.y_true = fct

    def set_training_parameters(self,input_list,output_list):
        params = self.meta

        self.weight_save_file = os.path.join(dasakl.DATA_FOLDER,'weights','{}_bc{}.hdf5')
        self.data = DataSet(input_list,output_list,first_shuffle=True)

        self.initial_learning_rate = params['learning_rate']
        self.decay_steps = params['decay_steps']
        self.decay_rate = params['decay_rate']
        self.N_epochs = params['N_epochs']
        self.batch_size = params['batch_size']
        self.save_epoch = params['save_epoch']

        self.global_step = tf.get_variable('global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        self.learning_rate =params['learning_rate']
        self.loss = self.get_loss(self.y,self.y_true)
        self.accuracy = self.get_accuracy(self.y,self.y_true)
        self.optimizer = optimizer[params['optimizer']](self.learning_rate)
        gradients = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(gradients,global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())

        self.logger= Log(os.path.join(dasakl.DATA_FOLDER,'log',self.name+'.log'),[
                'epoch  \t avg_cost \t avg_acc \t time',
                '{0:04d}  \t {1:.5e} \t {2:.5e} \t {3:06.2f} '
                            ])
        self.logger.header(description=['detector',self.name],model_text='''
                dimension: {} x {} x {} 
                training: {} epochs, {} batches
                initial lr: {}
                output: {} 
                '''.format(
                    self.meta['inp_size'][0],self.meta['inp_size'][1],self.meta['inp_size'][2],
                    self.N_epochs,self.batch_size,
                    self.initial_learning_rate,
                    self.weight_save_file))



    def detect(self,inputs):
        net_output = self.sess.run(self.y, feed_dict={self.x: inputs})
        return net_output


    def eval_layer(self,layer,image):
        preprocessed = self.preprocess(image,self.meta)
        if(isinstance(layer,type('strig'))):
            layer = [layer]
        layer_input = [self.model.layers[x] for x in layer]
        return self.sess.run(layer_input,feed_dict={self.x: preprocessed})

    def eval_loss(self,inp,target):
        preprocessed = self.preprocess(inp,self.meta)
        tar = self.read_output(target,self.meta)
        res = self.get_loss(self.y,self.y_true,self.meta)
        out = self.sess.run(res,feed_dict={self.x:preprocessed,self.y_true:tar})
        return out

    def analyze_gradients(self,feed_dict):
        grad = self.sess.run([self.gradients],feed_dict=feed_dict)[0]
        all_var = tf.trainable_variables()
        
        print("\t-- GRAD --")
        for idx in range(len(grad)):
            var = grad[idx][0]
            
            print('\t- mean {2:+010.8e} -mean(abs) {3:+010.8e} - min/max {4:+010.8e} / {5:+010.8e} - {0:>25} - {1:>15}'.format(all_var[idx].name,str(var.shape),var.mean(),abs(var).mean(),var.min(),var.max()))
        print("\t-- END GRAD --")
        return grad
        
    def image_detector(self, image,meta=None):
        preprocessed = self.preprocess(image,self.meta)
        net_out = self.detect(preprocessed)
        processed = self.postprocess(net_out,self.meta)

        return processed

    def camera_detector(self, camera, wait=10):
        preprocess_timer = Timer()
        network_timer = Timer()
        postprocess_timer = Timer()
        draw_timer = Timer()
        frame_timer = Timer()
        print('preprocess \t network \t postprocess \t draw \t total')
        count = 0
        while camera.isOpened():
            count += 1
            _, frame = camera.read()
            frame_timer.tic()

            preprocess_timer.tic()
            preprocessed = self.preprocess(frame,self.meta)
            preprocess_timer.toc()

            network_timer.tic()            
            net_out = self.detect(preprocessed)
            network_timer.toc()

            postprocess_timer.tic()
            processed = self.postprocess(net_out,self.meta)
            postprocess_timer.toc()

            draw_timer.tic()
            # processed = self.draw_result(frame,processed)
            draw_timer.toc()
            if(isinstance(processed,type([1,2,3]))):
                count = 0
                for im in processed:
                    count += 1
                    cv2.imshow('output {}'.format(count), im[0,:,:,:])    
            else:            
                cv2.imshow('Camera', processed[0,:,:,:])

            frame_timer.toc()

            print('{0:04.4f} \t {1:04.4f} \t {2:04.4f} \t {3:04.4f} \t {4:04.4f} \t '.format(
                    preprocess_timer.diff,network_timer.diff,postprocess_timer.diff,draw_timer.diff,frame_timer.diff))


            if cv2.waitKey(1) & 0xFF == ord('q'):
                camera.release()
                break
        camera.release()
        cv2.destroyAllWindows()

    
    def train(self):
        train_timer = Timer()
        load_timer = Timer()
        epoch_timer = Timer()
        batch_timer = Timer()
        # print("trainable variables")
        # print([var.name for var in tf.trainable_variables()])
        # params['batch_size'] = 64
        # params['N_epoch'] = 1000

        N_samples = self.data.N_samples
        N_batch = int(N_samples/self.batch_size)

        step = -1
        N_back_up = 0
        loss = 0

        self.sess.close()
        self.saver = tf.train.Saver() 
        sv = tf.train.Supervisor(logdir=self.session_path,
                                 # saver=self.saver,
                                 # save_model_secs=opt.save_interval,
                                 # summary_writer=summary_writer,
                                 # save_summaries_secs=opt.log_interval,
                                 # global_step=tf.sg_global_step(),
                                 # local_init_op=tf.sg_phase().assign(True)
                                 )
        with sv.managed_session() as sess:
        # self.sess.close()
        # sv = tf.train.Supervisor(logdir=self.session_path,
        #     save_model_secs=0)
        # self.sess = sv.managed_session()


            ### Loop for each epoch
            for epoch in range(1, self.N_epochs + 1):
                ### Start the timers
                epoch_loss = 0
                epoch_acc = 0
                epoch_load_diff = 0
                epoch_timer.tic()
                # load_timer.tic()

                ### Loop for the batches
                # A = self.sess.run('3-convolutional/kernel/Adagrad:0',feed_dict={self.x:self.preprocess('/home/gpu/data/shopkins/img80/000040003.png',self.meta)})
                # B = self.sess.run('2-convolutional/kernel:0',feed_dict={self.x:self.preprocess('/home/gpu/data/shopkins/img80/000040003.png',self.meta)})
                # print(A.shape,A.mean())
                # print(B.shape,B.mean())
                for batch in range(N_batch):
                    step += 1
                    load_timer.tic()

                    ### Get next batches as a path in general
                    inputs_path, outputs_path = self.data.next_batch(self.batch_size)

                    ### Convert the path into the appropriate format for the neural net
                    # print(inputs_path)
                    # print(outputs_path)
                    X_batch = np.concatenate([self.preprocess(x,self.meta) for x in inputs_path], axis=0)
                    Y_batch = np.concatenate([self.read_output(y,self.meta) for y in outputs_path], axis=0)

                    # arr = [self.preprocess(x,self.meta) for x in inputs_path]
                    # mmax = max([x.shape[1] for x in arr])
                    # dim2 = arr[0].shape[2]
                    # X_batch = np.concatenate([x.resize(1,mmax,dim2) for x in arr],axis=0)

                    # arr = [self.read_output(y,self.meta) for y in outputs_path]
                    # mmax = max([x.shape[1] for x in arr])
                    # Y_batch = np.concatenate([x.resize(1,mmax) for x in arr],axis=0)

                    # import dasakl.nn.vizualisation as viz
                    # print(inputs_path[0],np.unique(Y_batch[0]))
                    # viz.display(X_batch[0])

                    
                    # print(X_batch.shape,Y_batch.shape)
                    # label_flat = tf.reshape(self.y_true, (-1, 1))
                    # labels = tf.reshape(tf.one_hot(label_flat, depth=3), (-1, 3))

                    load_timer.toc()
                    epoch_load_diff += load_timer.diff
                    train_timer.tic()
                    # print(X_batch.shape,Y_batch.shape)
                    feed_dict = {self.x: X_batch, self.y_true: Y_batch}
                    

                    if(step%self.display_step==0):
                        print('\t Epoch {0}: {1}/{2} --acc = {3:09.1f} --loss = {4:09.4f} --train_time = {5:06.2f} --load_time = {6:06.2f}'.format(
                            epoch,batch+1,N_batch,epoch_acc/(batch+1),loss,train_timer.diff,load_timer.diff))
                        # A = self.sess.run([self.loss,self.y],feed_dict={self.x: X_batch[0:1], self.y_true: Y_batch[0:1]})
                        # print(A[0],A[1],Y_batch[0],X_batch[0].mean())
                        tmp = self.analyze_gradients(sess,feed_dict)


                    acc,loss,_ = sess.run([self.accuracy,self.loss, self.train_op],feed_dict=feed_dict)
                    # tmp,acc,loss,_ = self.sess.run([self.gradients,self.accuracy,self.loss, self.train_op],feed_dict=feed_dict)



                    
                    
                    # print(tmp[10])
                    # print(Y_batch[10])
                    # print(Y_batch[12,:])
                    # print(X_batch.shape,Y_batch)
                    # exit()
                    # print(acc.shape)
                    # exit()
                    # print(tmp.shape,tmp.dtype)
                    # print(np.unique(tmp[:,0]),tmp[:,0].mean())
                    # print(np.unique(tmp[:,1]),tmp[:,1].mean())
                    # print(np.unique(tmp[:,2]),tmp[:,2].mean())

                    epoch_loss += loss
                    epoch_acc += acc
                    train_timer.toc()
                    # print(acc)
                    # print(y[12,:])

                epoch_timer.toc()

                print('Epoch {0}: --acc = {1:09.4f} --loss = {2:09.4f} --time = {3:06.2f} --load time = {4:06.2f}'.format(
                            epoch,epoch_acc,epoch_loss,epoch_timer.diff,epoch_load_diff))
                self.logger.add(epoch,epoch_loss,epoch_acc,epoch_timer.diff)
                self.logger_last.add(epoch,epoch_loss,epoch_acc,epoch_timer.diff)

                # if(epoch%self.save_epoch==0):
                #     tmp = self.analyze_gradients(feed_dict)
                if(epoch%self.save_epoch==0):
                    N_back_up = np.mod(N_back_up+1,2)
                    file_save = self.weight_save_file.format(self.name,str(N_back_up))
                    f = h5py.File(self.weight_save_file.format(self.name,str(N_back_up)),'w')
                    # for key in self.var_dict.keys():
                        # A = self.sess.run(self.var_dict[key])
                    for key in [x.name for x in tf.global_variables()]:
                        A = sess.run(key)
                        # print(key,A.mean())
                        f.create_dataset(key,data=A)
                    f.close()
                    session_save = self.session_save_file.format(self.name,str(N_back_up))
                    self.saver.save(sess, session_save)
                    print('Weight file saved: {}'.format(file_save))
                    print('Session file saved: {}'.format(session_save))
        return 0

    def batch_generator(self,**kwargs):
        

        source = kwargs['source']
        dtypes = kwargs['dtypes']
        out_dtypes = dtypes
        capacity = kwargs['capacity']
        num_threads = kwargs['num_threads']
        def func(src_list):
            mfcc_file,label = src_list

            tmp = mfcc_file.decode('utf-8').split("/")[-1]
            label = self.read_output(label.decode('utf-8'),self.meta)[0]

            mfcc = self.preprocess(mfcc_file.decode('utf-8'),self.meta)[0]

            return mfcc, label
        def enqueue_func(sess, op):
            # read data from source queue

            data = func(src_list=sess.run(source))
            # create feeder dict
            feed_dict = {}
            for ph, col in zip(placeholders, data):
                feed_dict[ph] = col
            # run session
            sess.run(op, feed_dict=feed_dict)


        # create place holder list
        placeholders = []
        for dtype in dtypes:
            placeholders.append(tf.placeholder(dtype=dtype))

        
        # create FIFO queue
        fifo_queue = tf.FIFOQueue(capacity, dtypes=out_dtypes)

        # enqueue operation
        enqueue_op = fifo_queue.enqueue(placeholders)

        # create queue runner
        runner = queue._FuncQueueRunner(enqueue_func, fifo_queue, [enqueue_op] * num_threads)

        # register to global collection
        tf.train.add_queue_runner(runner)
        
        return fifo_queue.dequeue()

    def advanced_training_init(self,input_list,output_list):
        # N_samples = self.data.N_samples



        params = self.meta
        self.inputs = input_list
        self.targets = output_list




        prep = self.preprocess(self.inputs[0],self.meta)
        read = self.read_output(self.targets[0],self.meta)
        input_type = tf.float32 if 'float' in str(prep.dtype) else tf.int64
        input_shape = tuple(params['inp_size'])
        target_type = tf.float32 if 'float' in str(read.dtype) else tf.int64
        target_shape = tuple([int(x) for x in str(params['target_shape']).split(',')])
        target_shape = [x for x in target_shape if x != 0]
        target_shape = [None if x==-1 else x for x in target_shape]
        input_shape = [x for x in input_shape if x != 0]
        input_shape = [None if x==-1 else x for x in input_shape]


        self.batch_size = params['batch_size']
        self.N_samples = len(self.inputs)
        self.N_batch = int(self.N_samples/self.batch_size)

        label_t = tf.convert_to_tensor(self.targets)
        mfcc_file_t = tf.convert_to_tensor(self.inputs)
        mfcc_file_q, label_q = tf.train.slice_input_producer([mfcc_file_t,label_t], shuffle=True)


        capacity = 128
        num_threads = 2
        capacity_batch = self.batch_size*2
        capacity_batch = 128
        mfcc_q,label_q = self.batch_generator(source=[mfcc_file_q, label_q ],
                        dtypes=[input_type, target_type],
                        capacity=capacity, num_threads=num_threads)



        # create batch queue with dynamic pad
        batch_queue = tf.train.batch([mfcc_q, label_q ], self.batch_size,
                                     shapes=[input_shape, target_shape],
                                     num_threads=8, capacity=capacity_batch,
                                     dynamic_pad=True)


        


        self.x, self.y_true = batch_queue
    
    def train2(self):
        train_timer = Timer()
        load_timer = Timer()
        epoch_timer = Timer()
        batch_timer = Timer()

        step = -1
        N_back_up = 0

        self.sess.close()
        coord = tf.train.Coordinator()
        self.saver = tf.train.Saver() 

        sv = tf.train.Supervisor(logdir=self.session_path,
                                 # saver=self.saver,
                                 # save_model_secs=opt.save_interval,
                                 # summary_writer=summary_writer,
                                 # save_summaries_secs=opt.log_interval,
                                 # global_step=tf.sg_global_step(),
                                 # local_init_op=tf.sg_phase().assign(True)
                                 )
        with sv.managed_session(config=self.tf_config) as sess:
            train_timer.tic()
            try:
                # start queue thread
                threads = tf.train.start_queue_runners(sess, coord)
                
                for epoch in range(1, self.N_epochs + 1):
                    epoch_timer.tic()
                    loss_avg = 0.
                    epoch_loss = 0
                    epoch_acc = 0
                    for batch in range(self.N_batch):
                        batch_timer.tic()
                        step = step + 1
                        # run session
                        train_timer.tic()
                        acc, loss, _ = sess.run([self.accuracy,self.loss, self.train_op])
                        train_timer.toc()
                        epoch_loss += np.sum(loss)
                        epoch_acc += np.sum(acc)
                        # loss history update
                        # if batch_loss is not None and \
                        #         not np.isnan(batch_loss.all()) and not np.isinf(batch_loss.all()):
                        #     loss_avg += np.mean(batch_loss)

                        batch_timer.toc()
                        if(step%self.display_step==0):
                            print('\t Epoch {0}: {1}/{2} --acc = {3:09.1f} --loss = {4:09.4f} --batch_time = {5:06.2f} --batch_time = {5:06.2f}'.format(
                                epoch,batch+1,self.N_batch,epoch_acc/(batch+1),np.mean(loss),batch_timer.diff,train_timer.diff))
                    epoch_timer.toc()
                    
                    print('Epoch {0}: --loss = {1:09.4f} --time = {2:06.2f}'.format(
                            epoch,epoch_loss/(self.N_batch*self.batch_size),epoch_timer.diff))
                    # print('{} - Testing finished on {}.(CTC loss={})'.format(epoch,'TRAIN', loss_avg))
                    self.logger.add(epoch,epoch_loss,epoch_acc,epoch_timer.diff)
                    self.logger_last.add(epoch,epoch_loss,epoch_acc,epoch_timer.diff)
                    # final average
                    loss_avg /= self.N_batch * 1
                    if(epoch%self.save_epoch==0):
                        N_back_up = np.mod(N_back_up+1,2)
                        file_save = self.weight_save_file.format(self.name,str(N_back_up))
                        f = h5py.File(self.weight_save_file.format(self.name,str(N_back_up)),'w')
                        for key in self.var_dict.keys():
                            A = sess.run(self.var_dict[key])
                            # print(key,A.mean())
                            f.create_dataset(key,data=A)
                        f.close()
                        session_save = self.session_save_file.format(self.name,str(N_back_up))
                        self.saver.save(sess, session_save)
                        print('Weight file saved: {}'.format(file_save))
                        print('Session file saved: {}'.format(session_save))
                    
            finally:
                # stop queue thread
                train_timer.toc()
                coord.request_stop()
                # wait thread to exit.
                coord.join(threads)


def main():
    print('Not implemented')

if __name__ == '__main__':
    main()