from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.linalg as nl
from scipy import misc
import imageio

import matplotlib.pyplot as plt

import tensorflow as tf

imageToFloat = lambda x: x.astype(np.float64, copy=False)/255

def imshow(im):
    fig = plt.figure()
    plt.imshow(im)
    plt.show()

def imshow0(im):
    fig = plt.figure()
    plt.imshow(im)

def imageFromPixel(pixel, h, w):
    image = np.zeros((h,w,1), dtype=np.float32)
    return image+np.array(pixel, dtype=np.float32).reshape((1,1,3))

length = lambda x: np.prod(x.shape)

def colorDists(im1, im2):
    return np.sum((im1 - im2)**2, axis=2)**(1/2)

def partOfPixelsAroundGiven(pixel, radius, image):
    h, w, c = np.shape(image)
    imp = imageFromPixel(pixel, h, w)
    #dists = np.apply_along_axis(nl.norm, 2, (image - imp))
    dists =  colorDists(image, pixel)
    return length(dists[dists<radius])/length(dists)
    #return dists

def incrVar(p, factor):
    mean = np.mean(p)
    p = (p-mean)*factor + mean
    bound = lambda x: max(min(x,1.0), 0.0)
    return np.vectorize(bound)(p)

def colorToPdfInd(c):
    c = c*255
    i,j,k = np.vectorize(lambda x: int(round(x)))(c)
    return i,j,k

def colorPdf(image):
    cs = np.zeros((256,256,256), dtype=np.float64)
    for (i0,j0), value in np.ndenumerate(image[:,:,0]):
        cs[colorToPdfInd(image[i0,j0,:])] += 1
    return cs

def applyTransform(image, transMatrix, im2Pdf):
    out = np.zeros(image.shape, dtype=np.int32)
    for (i0,j0), value in np.ndenumerate(image[:,:,0]):
        out[i0,j0,:] = np.unravel_index(transMatrix[colorToPdfInd(image[i0,j0,:])], (256,256,256))
    #return imageToFloat(image)
    return imageToFloat(out)

def colorPdfTransform(im1, im2):
    n = 256
    #transMatrix = np.zeros((256,256,256), dtype=np.float64)
    transMatrix = np.zeros(np.prod([n,n,n]), dtype=np.int32)
    im1Pdf = colorPdf(im1).flatten()
    im2Pdf = colorPdf(im2).flatten()
    im1Sort = np.argsort(im1Pdf)
    im2Sort = np.argsort(im2Pdf)
    transMatrix[im1Sort] = im2Sort
    transMatrix = transMatrix.reshape(n,n,n)
    return applyTransform(im1, transMatrix, im2Pdf)

def incrColorful(image, factor):
    h, w, c = np.shape(image)
    mean = np.mean(image, axis=2).reshape(h,w,1)
    image = (image - mean)*factor + mean
    bound = lambda x: max(min(x,1.0), 0.0)
    return np.vectorize(bound)(image)

def ims():
    filenames = ['data/cur/1.jpg', 'data/cur/0.jpg']
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    ims = tf.image.decode_jpeg(value, channels=3)
    #ims = tf.image.rgb_to_grayscale(ims)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=sess)

    alice   = ims.eval(session=sess)
    reality = ims.eval(session=sess)

    #fig = plt.figure()
    #plt.imshow(alice)
    #fig = plt.figure()
    #plt.imshow(reality)
    #plt.show()

    return imageToFloat(alice), imageToFloat(reality)

alice, reality = ims()
n = 5
a = imageFromPixel([1,0,0],n,n)
b = imageFromPixel([1,1,0],n,n)
a[1,:,:]   = [0,0,0]
a[2:3,:,:] = [0,1,0]
b[0,:,:] = [1,1,1]
b[4:5,:,:] = [1,0,1]
a = reality
b = alice
imshow0(a)
imshow0(b)

#c = np.copy(b)

c0 = colorPdfTransform(a,b)
imshow(c0)

