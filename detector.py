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
from dasakl.nn.parser import get_layers,blm
from dasakl.utils import Timer, DataSet, Log
import os
import h5py



class Detector(object):
    def __init__(self, cfg, weights=None, GPU=True):
        self.name = os.path.split(cfg)[1].split('.')[0]
        tf.reset_default_graph()
        self.model = mmodel()
        self.meta,self.layers = self.model.parse(cfg)

        self.model.set_input(self.meta['inp_size'])

        self.x,self.y,self.var_dict = self.model.get_model(self.layers)

        for lay in list(self.model.layers.keys()):
            print(lay,self.model.layers[lay].shape)
        # exit()
        if(GPU):
            tf_config = tf.ConfigProto(
                log_device_placement=False,gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.80))
                        )
        else:
            tf_config = tf.ConfigProto(
                log_device_placement=False,device_count = {'GPU': 0})

        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        
        
        if(not isinstance(weights,type(None))):
            self.model.load_weights(self.sess,weights)

    def load_weights(self,filename,layers=None):
        self.model.load_weights(self.sess,filename,layers)

    def save_graph(self,filename):
        tf.train.write_graph(self.sess.graph_def, '/home/gpu/data', 'graph_segface.pb', as_text=False)
        
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
            # choice = cv2.waitKey(1)
            # if choice == 27: 
            #     camera.release()
            #     break
        camera.release()
        cv2.destroyAllWindows()

    
    def train(self):
        train_timer = Timer()
        load_timer = Timer()
        epoch_timer = Timer()
        batch_timer = Timer()
    
        N_samples = self.data.N_samples
        N_batch = int(N_samples/self.batch_size)

        step = 0
        N_back_up = 0
        loss = 0
        ### Loop for each epoch
        for epoch in range(1, self.N_epochs + 1):
            ### Start the timers
            epoch_loss = 0
            epoch_acc = 0
            epoch_timer.tic()
            load_timer.tic()

            ### Loop for the batches
            for batch in range(N_batch):
                step += 1
                load_timer.tic()

                ### Get next batches as a path in general
                inputs_path, outputs_path = self.data.next_batch(self.batch_size)


                ### Convert the path into the appropriate format for the neural net
                X_batch = np.concatenate([self.preprocess(x,self.meta) for x in inputs_path], axis=0)
                Y_batch = np.concatenate([self.read_output(y,self.meta) for y in outputs_path], axis=0)
                # print(Y_batch[12,:])

                label_flat = tf.reshape(self.y_true, (-1, 1))
                labels = tf.reshape(tf.one_hot(label_flat, depth=3), (-1, 3))

                load_timer.toc()

                train_timer.tic()
                feed_dict = {self.x: X_batch, self.y_true: Y_batch}
                
                tmp,acc,loss,_ = self.sess.run([labels,self.accuracy,self.loss, self.train_op],feed_dict=feed_dict)

                epoch_loss += loss
                epoch_acc += acc
                train_timer.toc()
                if(step%100==0):
                    print('\t Epoch {0}: {1}/{2} --acc = {3:09.1f} --loss = {4:09.4f} --train_time = {5:06.2f} --load_time = {6:06.2f}'.format(
                        epoch,batch+1,N_batch,epoch_acc/batch+1,loss,train_timer.diff,load_timer.diff))

            epoch_timer.toc()
            print('Epoch {0}: --acc = {1:09.4f} --loss = {2:09.4f} --time = {3:06.2f}'.format(
                        epoch,epoch_acc,epoch_loss,epoch_timer.diff))
            self.logger.add(epoch,epoch_loss,epoch_acc,epoch_timer.diff)

            if(epoch%self.save_epoch==0):
                N_back_up = np.mod(N_back_up+1,2)
                file_save = self.weight_save_file.format(self.name,str(N_back_up))
                f = h5py.File(self.weight_save_file.format(self.name,str(N_back_up)),'w')
                for key in self.var_dict.keys():
                    A = self.sess.run(self.var_dict[key])
                    f.create_dataset(key,data=A)
                f.close()
                print('Weight file saved: {}'.format(file_save))
        return 0



    

def main():
    print('Not implemented')

if __name__ == '__main__':
    main()