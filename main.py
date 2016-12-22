import os
import pickle
from utils import *
import h5py

with open("list_files.pkl", "rb") as output:
    list_path,list_label = pickle.load(output)

weight_folder = 'weights'
initial_weights = 'foo.hdf5'
initial_weights = None

N_classes = 3
dico_by_name = {}
dico_by_label = {}
dico_by_name['neutral'] = 0
dico_by_name['glasses'] = 1
dico_by_name['paci'] = 2

for key in dico_by_name.keys():
    tmp = dico_by_name[key]
    dico_by_label[tmp] = key


ds = DataSet(list_path,list_label,first_shuffle=True)

path = ds.next_batch(1)[0][0]



def transform_img(path):
    img = load_img(path,target_size=[28,28])
    img = img_to_array(img)
    img = preprocess_img(img)
    return img

img1 = transform_img(path)
import model
import tensorflow as tf


myModel = model.tf_cam(3)
x,y,var_dict = myModel.get_training_model()
# x,y,params = model.get_training_model()

y_  = tf.placeholder(tf.float32, [None, N_classes])

loss = myModel.get_loss(y,y_)
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)


for name in var_dict.keys():
    print(name,var_dict[name].get_shape())
exit()
sess = tf.Session(
        config=tf.ConfigProto(
                log_device_placement=False,
                gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.89)
                    )))




if(initial_weights is not None):
    f = h5py.File(os.path.join(weight_folder,initial_weights),'r')
    for name in var_dict.keys():
        sess.run(var_dict[name].assign(f[name].value))
    f.close()

# for var in tf.all_variables():
#     A = sess.run(var)
#     print(var.name,A.mean())

sess.run(tf.initialize_all_variables())

A = sess.run(y,feed_dict={x:img1})
print(A.shape)






#     A = A+0.1
#     sess.run(idx.assign(A))
#     # sess.run(tf.initialize_all_variables())
# # for idx in params:
#     A=sess.run(idx)
#     print(A.mean(),A[0])
#     print('===============')

# print('==')
# for idx in tf.all_variables():
#     A=sess.run(idx)
#     print(idx.name,A.shape,A.mean())


# exit(0)
# print([sess.run(p) for p in layers])
# exit()

for v in tf.trainable_variables():
    print(v.name)
N_epoch = 10
batch_size = 32
N_samples = len(list_path)
N_batch = int(N_samples/batch_size)

for epoch in range(N_epoch):

    for batch in range(N_batch):
        X_batch = []
        Y_batch = []
        path_batch,name_batch = ds.next_batch(batch_size)

        for idx in range(path_batch.shape[0]):
            # print(X_batch[idx])
            X_batch.append(transform_img(path_batch[idx]))
            Z = np.zeros((1,N_classes))
            idx = dico_by_name[name_batch[idx]]
            Z[0,idx] = 1
            Y_batch.append(Z)


        X_batch = np.concatenate(X_batch, axis=0)
        Y_batch = np.concatenate(Y_batch, axis=0)

        B = sess.run(train_op,feed_dict={x:X_batch,y_:Y_batch})
    A = sess.run(y,feed_dict={x:X_batch})
    be,co = sess.run([best,correct],feed_dict={x:X_batch,y_:Y_batch})
    for name in var_dict.keys():
        A=sess.run(var_dict[name])
        print(name,A.mean())
    print('=============')
    # print(A)
    print(be)
    print(co)



f = h5py.File(os.path.join(weight_folder,'trained.hdf5'),'w')
for key in var_dict.keys():
    A = sess.run(var_dict[key])
    f.create_dataset(key,data=A)

f.close()

# for layer in layers:
#     res = sess.run(layer,feed_dict={x:X_batch})
#     print(res)


# print(A)
sess.close()

