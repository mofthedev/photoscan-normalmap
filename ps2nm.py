# %%

# NM1 Functions

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
from skimage.util import img_as_float, img_as_int, img_as_uint, img_as_bool, img_as_float64, img_as_float32, img_as_ubyte
from skimage import exposure

def img_read_rgb(file_name):    
    img = img_as_ubyte(mpimg.imread(file_name))
    img = np.array(img)
    img = img.astype(int)
    return img

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    grayint = np.array(gray, dtype=np.uint8)
    return img_as_ubyte(grayint)


def show(img, format=None, interp=False):
    interp_arg = 'antialiased'
    if interp == False:
        interp_arg = 'none'
    # fig = plt.figure()
    # fig.set_size_inches(10,10)
    plt.imshow(img, format, interpolation=interp_arg, )
    plt.show()

# source: https://stackoverflow.com/a/52143032
def overlay(a,b):
    a = a.astype(float)/255
    b = b.astype(float)/255 # make float on range 0-1

    mask = a >= 0.5 # generate boolean mask of everywhere a > 0.5 
    ab = np.zeros_like(a) # generate an output container for the blended image 
    
    # now do the blending 
    ab[~mask] = (2*a*b)[~mask] # 2ab everywhere a<0.5
    ab[mask] = (1-2*(1-a)*(1-b))[mask] # else this 
    return (ab*255).astype(np.uint8)


def equalize(img, clip_limit_ = 0.03):
    imgeq = exposure.equalize_adapthist(img, clip_limit=clip_limit_)
    imgeq = img_as_ubyte(imgeq)
    return imgeq






def ps2nm(imgU, imgD, imgR, imgL):
    # 1-channel empty image
    imgE = np.zeros_like(imgU)


    # combine L & U
    img_LU = np.stack((imgL,imgU, imgE), axis=2)
    # show(img_LU)
    img_LU = ((255-img_LU)/(255/127)).astype(np.uint8)

    # show(img_LU)
    # print("*"*30)


    # combine R & D
    img_RD = np.stack((imgR,imgD, imgE), axis=2)
    # show(img_RD)
    img_RD = ((img_RD/255)*127+128).astype(np.uint8)

    # show(img_RD)
    # print("*"*30)



    # overlay
    img_O = overlay(img_LU, img_RD)
    img_O[:,:,2] = img_O[:,:,2] + 127
    return img_O






# NM2 Functions

import scipy.ndimage
import scipy.misc
from scipy import ndimage

def smooth_gaussian(im, sigma):

    if sigma == 0:
        return im

    im_smooth = im.astype(float)
    kernel_x = np.arange(-3*sigma,3*sigma+1).astype(float)
    kernel_x = np.exp((-(kernel_x**2))/(2*(sigma**2)))

    im_smooth = scipy.ndimage.convolve(im_smooth, kernel_x[np.newaxis])

    im_smooth = scipy.ndimage.convolve(im_smooth, kernel_x[np.newaxis].T)

    return im_smooth


def gradient(im_smooth):

    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.arange(-1,2).astype(float)
    kernel = - kernel / 2

    gradient_x = scipy.ndimage.convolve(gradient_x, kernel[np.newaxis])
    gradient_y = scipy.ndimage.convolve(gradient_y, kernel[np.newaxis].T)

    return gradient_x,gradient_y


def sobel(im_smooth):
    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    gradient_x = scipy.ndimage.convolve(gradient_x, kernel)
    gradient_y = scipy.ndimage.convolve(gradient_y, kernel.T)

    # show(gradient_x)
    # show(gradient_y)

    return gradient_x,gradient_y


def compute_normal_map(gradient_x, gradient_y, intensity=1):

    width = gradient_x.shape[1]
    height = gradient_x.shape[0]
    max_x = np.max(gradient_x)
    max_y = np.max(gradient_y)

    max_value = max_x

    if max_y > max_x:
        max_value = max_y

    normal_map = np.zeros((height, width, 3), dtype=np.float32)

    intensity = 1 / intensity

    strength = max_value / (max_value * intensity)

    normal_map[:,:, 0] = gradient_x / max_value
    normal_map[:,:, 1] = gradient_y / max_value
    normal_map[:,:, 2] = 1 / strength

    norm = np.sqrt(np.power(normal_map[:,:, 0], 2) + np.power(normal_map[:,:, 1], 2) + np.power(normal_map[:,:, 2], 2))

    normal_map[:,:, 0] /= norm
    normal_map[:,:, 1] /= norm
    normal_map[:,:, 2] /= norm

    normal_map *= 0.5
    normal_map += 0.5

    return normal_map

#heightmap to normalmap
def hm2nm(img, normalIntensity = 4.0, smoothness = 0.1):
    im_smooth = smooth_gaussian(img, smoothness)

    sobel_x, sobel_y = sobel(im_smooth)

    normal_map = compute_normal_map(sobel_x, sobel_y, normalIntensity)

    normal_map = (normal_map*255).astype(np.uint8)
    return normal_map




print("functions defined")





# %%
# Application





# NM1: photometric stereo
imgU = rgb2gray(img_read_rgb("up.jpg"))
imgD = rgb2gray(img_read_rgb("down.jpg"))
imgR = rgb2gray(img_read_rgb("right.jpg"))
imgL = rgb2gray(img_read_rgb("left.jpg"))


img_NM1 = ps2nm(imgU,imgD,imgR,imgL)
print("photoscan to normalmap")
show(img_NM1, interp=True)





# NM2: heightmap to normalmap
imgAvg1 = overlay(equalize(imgU),equalize(imgD))
imgAvg2 = overlay(equalize(imgR),equalize(imgL))
imgAvg = equalize(overlay(imgAvg1,imgAvg2))
# print("combined grays")
# show(imgAvg, 'gray')

img_NM2 = hm2nm(imgAvg, 4, 0.1)
print("normalmap from combined grays")
show(img_NM2,interp=True)







# NM_Combined: Overlay-blending two normalmaps generated using both methods
img_NM_Combined = overlay(img_NM1 , img_NM2)
print("combination of two normalmaps")
show(img_NM_Combined, interp=True)


# other examples without photometric stereo images
show(hm2nm(equalize(rgb2gray(img_read_rgb("img4.png"))),4,0.1),interp=True)
show(hm2nm(equalize(rgb2gray(img_read_rgb("img6.jpeg"))),4,0.1),interp=True)
