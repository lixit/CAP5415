import numpy as np
import cv2
from scipy import ndimage
from scipy import linalg
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import pdb 


def convolution(image, kernel):
    '''
    Performs convolution along x and y axis, based on kernel size.
    Assumes input image is 1 channel (Grayscale)
    Inputs: 
      image: H x W x C shape numpy array (C=1)
      kernel: K_H x K_W shape numpy array (for example, 3x1 for 1 dimensional filter for y-component)
    Returns:
      H x W x C Image convolved with kernel
    '''
    # Get kernel size
    k_h, k_w = kernel.shape
    # Get image size
    i_h, i_w, i_c = image.shape
    # Pad image with zeros on all sides
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded_image = np.zeros((i_h + pad_h*2, i_w + pad_w*2, i_c))
    padded_image[pad_h:-pad_h, pad_w:-pad_w, :] = image
    # Create output image
    output_image = np.zeros((i_h, i_w, i_c))
    # Convolve image with kernel
    for i in range(i_h):
        for j in range(i_w):
            for c in range(i_c):
                output_image[i, j, c] = np.sum(padded_image[i:i+k_h, j:j+k_w, c] * kernel)
    return output_image

def convolution_1d(image, kernel):
    '''
    Performs convolution along x and y axis, based on kernel size.
    Assumes input image is 1 channel (Grayscale)
    Inputs: 
      image: H x W x C shape numpy array (C=1)
      kernel: K_H x K_W shape numpy array (for example, 3x1 for 1 dimensional filter for y-component)
    Returns:
      H x W x C Image convolved with kernel
    '''
    # Get kernel size
    k_h, k_w = kernel.shape
    # Get image size
    i_h, i_w = image.shape
    # Pad image with zeros base on kernel size
    if k_h == 1:
        print("filter for x-component")
        pad_w = k_w // 2
        padded_image = np.zeros((i_h, i_w + pad_w*2))
        padded_image[ : , pad_w:-pad_w] = image
        # Create output image
        output_image = np.zeros((i_h, i_w))
        # Convolve image with kernel
        for i in range(i_h):
            for j in range(i_w):
                output_image[i, j] = np.sum(padded_image[i, j:j+k_w] * kernel)
    elif k_w == 1:
        print("filter for y-component")
        pad_h = k_h // 2
        padded_image = np.zeros((i_h + pad_h*2, i_w))
        padded_image[pad_h:-pad_h, : ] = image
        # Create output image
        output_image = np.zeros((i_h, i_w))
        # Convolve image with kernel
        for i in range(i_h):
            for j in range(i_w):
                output_image[i, j] = np.sum(padded_image[i:i+k_h, j] * kernel.reshape(k_h))
    
    return output_image



def gaussian_kernel(size=3, sigma=1):
    '''
    Creates Gaussian Kernel (1 Dimensional)
    Inputs: 
      size: width of the filter
      sigma: standard deviation
    Returns a 1xN shape 1D gaussian kernel
    '''
    # Create 1D Gaussian Kernel
    kernel_1d = np.zeros((1, size))
    kernel_1d[0, : ] = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        x = kernel_1d[0, i]
        kernel_1d[0, i] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.square(x / sigma))
    # Normalize kernel
    kernel_1d[0, : ] = kernel_1d[0, : ] / np.sum(kernel_1d[0, : ])
    return kernel_1d


def gaussian_first_derivative_kernel(size=3, sigma=1):
    '''
    Creates 1st derviative gaussian Kernel (1 Dimensional)
    Inputs: 
      size: width of the filter
      sigma: standard deviation
    Returns a 1xN shape 1D 1st derivative gaussian kernel
    '''
    # Create 1D Gaussian Kernel
    kernel_1d = np.zeros((1, size))
    kernel_1d[0,:] = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        x = kernel_1d[0, i]
        # Derivatives of Gaussian = -x / np.square(sigma) * Gaussian
        kernel_1d[0, i] = -x / np.square(sigma) * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.square(x / sigma))
    return kernel_1d


def non_max_supression(det, phase):
    '''
    Performs non-maxima supression for given magnitude and orientation.
    Returns output with nms applied. Also return a colored image based on gradient direction for maximum value.
    '''
    # Get image size
    i_h, i_w = det.shape
    # Create output image
    output_image = np.zeros((i_h, i_w))
    # Create colored image
    colored_image = np.zeros((i_h, i_w, 3))
    # Iterate over image
    for i in range(i_h):
        for j in range(i_w):
            # Get orientation
            theta = phase[i, j]
            # Get magnitude
            mag = det[i, j]
            # Get maximum value in 3x3 window
            max_val = np.max(det[max(i-1, 0):min(i+2, i_h), max(j-1, 0):min(j+2, i_w)])
            # If current pixel is maximum in window, keep it
            if mag == max_val:
                output_image[i, j] = mag
                # Color pixel based on orientation
                if theta >= 0 and theta < 45:
                    colored_image[i, j, 0] = 255
                elif theta >= 45 and theta < 90:
                    colored_image[i, j, 0] = 255
                    colored_image[i, j, 1] = 255
                elif theta >= 90 and theta < 135:
                    colored_image[i, j, 1] = 255
                elif theta >= 135 and theta < 180:
                    colored_image[i, j, 1] = 255
                    colored_image[i, j, 2] = 255
                elif theta >= 180 and theta < 225:
                    colored_image[i, j, 2] = 255
                elif theta >= 225 and theta < 270:
                    colored_image[i, j, 0] = 255
                    colored_image[i, j, 2] = 255
                elif theta >= 270 and theta < 315:
                    colored_image[i, j, 0] = 255
                elif theta >= 315 and theta < 360:
                    colored_image[i, j, 0] = 255
                    colored_image[i, j, 1] = 255
    return output_image, colored_image

def DFS(img):
    '''
    If pixel is linked to a strong pixel in a local window, make it strong as well.
    Called iteratively to make all strong-linked pixels strong.
    '''
    # Get image size
    i_h, i_w = img.shape
    # Iterate over image
    for i in range(i_h):
        for j in range(i_w):
            # If pixel is strong, check if it is linked to a strong pixel
            if img[i, j] == 255:
                # Check if pixel is in a local window of a strong pixel
                if np.max(img[max(i-1, 0):min(i+2, i_h), max(j-1, 0):min(j+2, i_w)]) == 255:
                    # If yes, make it strong
                    img[i, j] = 255
    return img
    

                    
def hysteresis_thresholding(img, low_ratio, high_ratio):
    '''
    Performs hysteresis thresholding for given image and low and high thresholds.
    Returns output with hysteresis thresholding applied.
    '''
    # Get image size
    i_h, i_w = img.shape
    # Get low and high thresholds
    low_threshold = np.max(img) * low_ratio
    high_threshold = np.max(img) * high_ratio
    # Create output image
    output_image = np.zeros((i_h, i_w))
    # Iterate over image
    for i in range(i_h):
        for j in range(i_w):
            # If pixel is strong, keep it
            if img[i, j] >= high_threshold:
                output_image[i, j] = 255
            # If pixel is weak, check if it is linked to a strong pixel
            elif img[i, j] >= low_threshold:
                # Check if pixel is in a local window of a strong pixel
                if np.max(img[max(i-1, 0):min(i+2, i_h), max(j-1, 0):min(j+2, i_w)]) >= high_threshold:
                    # If yes, make it strong
                    output_image[i, j] = 255
    # Perform DFS to make all strong-linked pixels strong
    output_image = DFS(output_image)
    return output_image

def main():
    # Initialize values
    # You can choose any sigma values like 1, 0.5, 1.5, etc
    sigma = 3


    # Read the image in grayscale mode using opencv
    I = cv2.imread(r'C:\Users\a\Downloads\cda5415\46076.jpg', cv2.IMREAD_GRAYSCALE)

    # Create a gaussian kernel 1XN matrix
    G = gaussian_kernel(size=3, sigma=sigma)


    # # Convolution of G and I
    # I_xx = convolution_1d(I, G)
    # I_yy = convolution_1d(I, G.T)
    # # plot the image using matplotlib
    # plt.subplot(121),plt.imshow(I_xx, cmap = 'gray')
    # plt.title('Gaussian along x Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(I_yy,cmap = 'gray')
    # plt.title('Gaussian along y Image'), plt.xticks([]), plt.yticks([])
    # plt.show()


    # Get the First Derivative Kernel
    G_x = gaussian_first_derivative_kernel(size=3, sigma=sigma)

    # Derivative of Gaussian Convolution
    I_x_prime = convolution_1d(I, G_x)
    I_y_prime = convolution_1d(I, G_x.T)    

    # Convert derivative result to 0-255 for display.
    # Need to scale from 0-1 to 0-255.
    #abs_grad_x = (( (I_xx - np.min(I_xx)) / (np.max(I_xx) - np.min(I_xx)) ) * 255.).astype(np.uint8)  
    #abs_grad_y = (( (I_yy - np.min(I_yy)) / (np.max(I_yy) - np.min(I_yy)) ) * 255.).astype(np.uint8)
    I_x_prime = (( (I_x_prime - np.min(I_x_prime)) / (np.max(I_x_prime) - np.min(I_x_prime)) ) * 255.).astype(np.uint8)
    I_y_prime = (( (I_y_prime - np.min(I_y_prime)) / (np.max(I_y_prime) - np.min(I_y_prime)) ) * 255.).astype(np.uint8)
    plt.subplot(121),plt.imshow(I_x_prime, cmap = 'gray')
    plt.title('derivative of Gaussian along x Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(I_y_prime, cmap = 'gray')
    plt.title('derivative of Gaussian along y Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Compute magnitude
    M = np.sqrt(np.square(I_x_prime) + np.square(I_y_prime))

    # Compute orientation
    O = np.arctan2(I_y_prime, I_x_prime) * 180 / np.pi
    O = O % 360
    print(O.shape)

    
    # Compute non-max suppression
    M_nms, O_nms = non_max_supression(M, O)
    # plot the image using matplotlib
    plt.subplot(121),plt.imshow(M_nms, cmap = 'gray')
    plt.title('Magnitude Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(O_nms, cmap = 'gray')
    plt.title('Orientation Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    #Compute thresholding and then hysteresis



    
if __name__ == '__main__':

    main()
    