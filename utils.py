import numpy as np
from PIL import Image
from collections import defaultdict


def get_dico(model=21):
    dico_by_name = defaultdict()
    dico_by_code = defaultdict()
    dico_by_label = defaultdict()

    if(model==21):
        dico_by_name['airplane'] = ['n02691156',0]
        dico_by_name['baseball'] = ['n02799071',1]
        dico_by_name['bear'] = ['n02131653',2]
        dico_by_name['bird'] = ['n01503061',3]
        dico_by_name['bowl'] = ['n02880940',4]
        dico_by_name['car'] = ['n02958343',5]
        dico_by_name['chair'] = ['n03001627',6]
        dico_by_name['mug'] = ['n03797390',7]
        dico_by_name['digital_clock'] = ['n03196217',8]
        dico_by_name['frog'] = ['n01639765',9]
        dico_by_name['goldfish'] = ['n01443537',10]
        dico_by_name['lemon'] = ['n07749582',11]
        dico_by_name['orange'] = ['n07747607',12]
        dico_by_name['pencil_case'] = ['n03908618',13]
        dico_by_name['pencil_sharpener'] = ['n03908714',14]
        dico_by_name['remote'] = ['n04074963',15]
        dico_by_name['sheep'] = ['n02411705',16]
        dico_by_name['sunglasses'] = ['n04356056',17]
        dico_by_name['tennis_ball'] = ['n04409515',18]
        dico_by_name['bottle'] = ['n04557648',19]
        dico_by_name['wine_bottle'] = ['n04591713',20]

    if(model==3):
        dico_by_name['neutral'] = ['neutral',0]
        dico_by_name['glasses'] = ['glasses',1]
        dico_by_name['paci'] = ['paci',2]

    if(model==32):
        dico = {}
        dico_by_label[0] = [[64, 128, 64], 'Animal']
        dico_by_label[1] = [[192, 0, 128], 'Archway']
        dico_by_label[2] = [[0, 128, 192], 'Bicyclist']
        dico_by_label[3] = [[0, 128, 64], 'Bridge']
        dico_by_label[4] = [[128, 0, 0], 'Building']
        dico_by_label[5] = [[64, 0, 128], 'Car']
        dico_by_label[6] = [[64, 0, 192], 'CartLuggagePram']
        dico_by_label[7] = [[192, 128, 64], 'Child']
        dico_by_label[8] = [[192, 192, 128], 'Column_Pole']
        dico_by_label[9] = [[64, 64, 128], 'Fence']
        dico_by_label[10] = [[128, 0, 192], 'LaneMkgsDriv']
        dico_by_label[11] = [[192, 0, 64], 'LaneMkgsNonDriv']
        dico_by_label[12] = [[128, 128, 64], 'Misc_Text']
        dico_by_label[13] = [[192, 0, 192], 'MotorcycleScooter']
        dico_by_label[14] = [[128, 64, 64], 'OtherMoving']
        dico_by_label[15] = [[64, 192, 128], 'ParkingBlock']
        dico_by_label[16] = [[64, 64, 0], 'Pedestrian']
        dico_by_label[17] = [[128, 64, 128], 'Road']
        dico_by_label[18] = [[128, 128, 192], 'RoadShoulder']
        dico_by_label[19] = [[0, 0, 192], 'Sidewalk']
        dico_by_label[20] = [[192, 128, 128], 'SignSymbol']
        dico_by_label[21] = [[128, 128, 128], 'Sky']
        dico_by_label[22] = [[64, 128, 192], 'SUVPickupTruck']
        dico_by_label[23] = [[0, 0, 64], 'TrafficCone']
        dico_by_label[24] = [[0, 64, 64], 'TrafficLight']
        dico_by_label[25] = [[192, 64, 128], 'Train']
        dico_by_label[26] = [[128, 128, 0], 'Tree']
        dico_by_label[27] = [[192, 128, 192], 'Truck_Bus']
        dico_by_label[28] = [[64, 0, 64], 'Tunnel']
        dico_by_label[29] = [[192, 192, 0], 'VegetationMisc']
        dico_by_label[30] = [[0, 0, 0], 'Void']
        dico_by_label[31] = [[64, 192, 0], 'Wall']


        dico_by_color = {}
        for idx in dico_by_label:
            A = dico_by_label[idx][0]
            d_name = dico_by_label[idx][1]
            dico_by_code['r{}g{}b{}'.format(A[0],A[1],A[2])] = [idx,dico_by_label[idx][1]]
            dico_by_name[d_name] = [idx,A]

    else:
        for key in dico_by_name.keys():
            tmp = dico_by_name[key]
            dico_by_label[tmp[1]] = key
            dico_by_code[dico_by_name[key][0]] = [key,dico_by_name[key][1]]
    return [dico_by_name,dico_by_code,dico_by_label]


def load_img(path, grayscale=False, target_size=None):

    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img

# load_img(os.path.join('data','n04409515','n04409515_488.JPEG'))



def img_to_array(img, dim_ordering='tf',expand=True):
    if dim_ordering == 'default':
        dim_ordering = 'th'
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    if(expand):
        x = np.expand_dims(x, axis=0)
    return x

def preprocess_img(x, dim_ordering='vgg', mean=[0.,0.,0.], rgb='rgb'):

    if dim_ordering == 'theano_vgg':
        x[:, 0, :, :] -= mean[0]
        x[:, 1, :, :] -= mean[1]
        x[:, 2, :, :] -= mean[2]
        # 'RGB'->'BGR'
        if(rgb.lower()=='bgr'):
            x = x[:, ::-1, :, :]
    elif dim_ordering == 'vgg':
        x[:, :, :, 0] -= mean[0]
        x[:, :, :, 1] -= mean[1]
        x[:, :, :, 2] -= mean[2]
        # 'RGB'->'BGR'
        # if(rgb.lower()=='bgr'):
        x = x[:, :, :, ::-1]
    else:
        x = (x-128.)/128.
    return x


class DataSet(object):
    def __init__(self, images, labels=None, fake_data=False, first_shuffle=False):
        # if fake_data:
        #     self._num_examples = 10000
        # else:
        #     if labels is not None:
        #         assert images.shape[0] == labels.shape[0], (
        #             "images.shape: %s labels.shape: %s" % (images.shape,
        #                                                    labels.shape))
        if(isinstance(images,type([]))):
            images = np.asarray(images)
        if(isinstance(labels,type([]))):
            labels = np.asarray(labels)

        self._num_examples = images.shape[0]
        # images = images.astype(np.float32)
        # images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        if(first_shuffle):
            self._index_in_epoch = self._num_examples + 1




    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        # print(self._index_in_epoch,self._epochs_completed,self._num_examples)
        return self._images[start:end], self._labels[start:end]



