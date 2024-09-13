import numpy as np
import cv2
from math import pi
import os
from random import randint
from PIL import Image
""" Utility Functions """

def load_image(img_path, shape=None):
    img = cv2.imread(img_path)
    if shape is not None:
        img = cv2.resize(img, shape)
    return img

def save_image(img_path, img):
    cv2.imwrite(img_path, img)

def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))

def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta),
            rad_to_deg(rphi),
            rad_to_deg(rgamma))

def deg_to_rad(deg):
    return deg * pi / 180.0

def rad_to_deg(rad):
    return rad* 180.0 / pi
# Usage: 
#     Change main function with ideal arguments
#     Then
#     from image_tranformer import ImageTransformer
#
# Parameters:
#     image_path: the path of image that you want rotated
#     shape     : the ideal shape of input image, None for original size.
#     theta     : rotation around the x axis
#     phi       : rotation around the y axis
#     gamma     : rotation around the z axis (basically a 2D rotation)
#     dx        : translation along the x axis
#     dy        : translation along the y axis
#     dz        : translation along the z axis (distance to the image)
#
# Output:
#     image     : the rotated image
# 
# Reference:
#     1.        : http://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles
#     2.        : http://jepsonsblog.blogspot.tw/2012/11/rotation-in-3d-using-opencvs.html


class ImageTransformer(object):
    """ Perspective transformation class for image
        with shape (height, width, #channels) """

    def __init__(self, image_path, shape):
        self.image_path = image_path
        self.image = load_image(image_path, shape)
 
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.num_channels = self.image.shape[2]


    """ Wrapper of Rotating a Image """
    def rotate_along_axis(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        
        # Get radius of rotation along 3 axes
        rtheta, rphi, rgamma = get_rad(theta, phi, gamma)
        
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height**2 + self.width**2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, dx, dy, dz)
        
        return cv2.warpPerspective(self.image.copy(), mat, (self.width, self.height))


    """ Get Perspective Projection Matrix """
    def get_M(self, theta, phi, gamma, dx, dy, dz):
        
        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))


def rotation(numero):
    path="panneau/"
    tab=os.listdir(path)
    choix=randint(0,len(tab)-1)
    path=path+"/"+str(tab[choix])
    name=numero+';'+str(tab[choix])[:-3]+'JPG'
    angle=randint(0,120)
    angle=angle-60
    rotation=ImageTransformer(path,None)
    rotated_img=rotation.rotate_along_axis(phi=angle)
    save_image('compresser/'+name+".JPG", rotated_img)
    imgComp=compresser('compresser/'+name,10)
    imgComp.save('output/'+name)
    
def cercle(numero):
    path="panneau/"
    tab=os.listdir(path)
    choix=randint(0,len(tab)-1)
    path=path+"/"+str(tab[choix])
    name=numero+';'+str(tab[choix][:-4])
    img=Image.open(path)
    largeur,hauteur=img.size
    rayon=8
    centre_l=randint(0,largeur)
    centre_h=randint(0,hauteur)
    marron=(88,41,0)
    for k in range(rayon*2):
        for l in range(rayon*2):
            i=k-rayon
            j=l-rayon
            if (centre_l+i >0 and centre_l+i<largeur and centre_h+j>0 and centre_h+j<hauteur):
                if (i**2+j**2<rayon**2 ):
                    img.putpixel((centre_l+i,centre_h+j),marron)
    imgComp=img.resize((10,10))
    imgComp=imgComp.convert('RGB')
    imgComp.save('output/'+name+'.JPG')
  

def create_data(c,r,taille):
    for k in range(taille):
        if (c and r):
            cercle(str(2*k))
            rotation(str(2*k+1))
        elif c:
            cercle(str(k))
        elif r:
            rotation(str(k))
def delete_data():
  dir=["aggrandissement/","compresser/","output/"]
  tab=[os.listdir("aggrandissement/"),os.listdir("compresser/"),os.listdir("output/")]
  for k in range(3):
    for i in range(len(tab[k])-1):
      if not(os.path.isdir(dir[k]+tab[k][i+1])):
        os.remove(dir[k]+tab[k][i+1])

def compresser(path, taille):
  try:
    img=Image.open(path+'.JPG')
  except :
    img=Image.open(path[:-4]+'.JPG')
  imgComp=img.resize((taille,taille))
  return imgComp

delete_data()
create_data(True,True,5)