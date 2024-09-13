import os
import random
from random import randint
from PIL import Image
import numpy as np
import scipy

def random_rotation_3d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    batch of rotated 3D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image1 = np.squeeze(batch[i])
            # rotate along z-axis
            angle = random.uniform(-max_angle, max_angle)
            image2 = scipy.ndimage.interpolation.rotate(image1, angle, mode='nearest', axes=(0, 1), reshape=False)

            # rotate along y-axis
            angle = random.uniform(-max_angle, max_angle)
            image3 = scipy.ndimage.interpolation.rotate(image2, angle, mode='nearest', axes=(0, 2), reshape=False)

            # rotate along x-axis
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image3, angle, mode='nearest', axes=(1, 2), reshape=False)
            #                print(i)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)

choix=randint(1,6)
path="panneau/"+str(choix)
tab=os.listdir(path)
choix=randint(0,len(tab)-1)
path=path+"/"+str(tab[choix])
img=Image.open(path)
pic = np.asarray(img)
random_rotation_3d(pic,60)
