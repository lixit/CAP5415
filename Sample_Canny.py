import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

def convolution_1d(image, kernel):
    '''
    Performs convolution along x or y axis, based on kernel size.
    Assumes input image is 1 channel (Grayscale)
    Inputs: 
      image: H x W shape numpy array
      kernel: K_H x K_W shape numpy array (for example, 3x1 for 1 dimensional filter for y-component)
    Returns:
      H x W Image convolved with kernel
    '''
    # Get kernel size
    k_h, k_w = kernel.shape
    # Get image size
    i_h, i_w = image.shape
    # filter for x-component
    if k_h == 1:
        # Pad image with zeros on left and right
        pad_w = k_w // 2
        padded_image = np.zeros((i_h, i_w + pad_w*2))
        padded_image[ : , pad_w:-pad_w] = image
        # Create output image
        output_image = np.zeros((i_h, i_w))
        # Convolve image with kernel
        for i in range(i_h):
            for j in range(i_w):
                output_image[i, j] = np.sum(padded_image[i, j:j+k_w] * kernel)
    # filter for y-component
    elif k_w == 1:
        # Pad image with zeros on up and down
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
    # Create 1*N Gaussian Kernel
    kernel_1d = np.zeros((1, size))
    # Fill the first row with values. e.g. [-1, 0, 1]
    kernel_1d[0, : ] = np.linspace(-(size // 2), size // 2, size)
    # Discritize the gaussian function
    for i in range(size):
        x = kernel_1d[0, i]
        # The actual gaussian function
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
    # Create 1*N matrix filled with zeros
    kernel_1d = np.zeros((1, size))
    kernel_1d[0,:] = np.linspace(-(size // 2), size // 2, size)
    # Discritize the drivative of gaussian function with e.g. [-1, 0, 1]
    for i in range(size):
        x = kernel_1d[0, i]
        # Derivatives of Gaussian = -x / np.square(sigma) * Gaussian
        kernel_1d[0, i] = -x / np.square(sigma) * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * np.square(x / sigma))
    return kernel_1d


def non_max_suppression(magnitude, ori):
    '''
    Performs non-maxima suppression for given magnitude and orientation.
    Inputs: 
      magnitude: H x W shape numpy array
      ori: in radians, in the range [-pi, pi]
    Return:
        output with NMS applied.
    '''
    # Get image size
    i_h, i_w = magnitude.shape
    # convert orientation from radians to degrees [-180, 180]
    angle = ori * 180. / np.pi
    # cast to [0, 180], e.g. -315 -> 45. Only the line's orientation is enough
    angle[angle < 0] += 180

    # Create output image
    output_image = np.zeros((i_h, i_w))

    # find the max value in 3x3 window at current pixel's orientation
    # be careful: the y axis is downward.
    #  0 ---------> x
    #  |
    #  |
    #  V
    #  y
    for i in range(i_h):
        for j in range(i_w):
            # 8 directions to go
            up = 0 if i + 1 >= i_h else magnitude[i+1, j]
            down = 0 if i - 1 < 0 else magnitude[i-1, j]
            left = 0 if j - 1 < 0 else magnitude[i, j-1]
            right = 0 if j + 1 >= i_w else magnitude[i, j+1]
            up_left = 0 if i - 1 < 0 or j - 1 < 0 else magnitude[i-1, j-1]
            up_right = 0 if i - 1 < 0 or j + 1 >= i_w else magnitude[i-1, j+1]
            down_left = 0 if i + 1 <= i_h or j - 1 < 0 else magnitude[i+1, j-1]
            down_right = 0 if i + 1 >= i_h or j + 1 >= i_w else magnitude[i+1, j+1]

            theta = angle[i, j]
            if theta < 45:
                mag1 = theta / 45 * down_right + (45 - theta) / 45 * right
                mag2 = theta / 45 * up_left + (45 - theta) / 45 * left
                
            elif theta >= 45 and theta < 90:
                thera_percent = (theta - 45) / 45
                mag1 = thera_percent * down + (1 - thera_percent) * down_right
                mag2 = thera_percent * up + (1 - thera_percent) * up_left

            elif theta >= 90 and theta < 135:
                thera_percent = (theta - 90) / 45
                mag1 = thera_percent * down_left + (1 - thera_percent) * down
                mag2 = thera_percent * up_right + (1 - thera_percent) * up

            else: # theta >= 135
                thera_percent = (theta - 135) / 45
                mag1 = thera_percent * left + (1 - thera_percent) * down_left
                mag2 = thera_percent * right + (1 - thera_percent) * up_right
            
            mag = magnitude[i, j]
            if mag > mag1 and mag > mag2:
                output_image[i, j] = mag

    return output_image

def non_max_suppression2(magnitude, ori):
    '''
    Old version, use 4 directions and no interpolation
    '''
    # Get image size
    i_h, i_w = magnitude.shape
    # convert orientation from radians to degrees [-180, 180]
    ori = ori * 180. / np.pi
    # cast to [0, 180], e.g. -315 -> 45. Only the line's orientation is enough
    ori[ori < 0] += 180

    # Create output image
    output_image = np.zeros((i_h, i_w))

    # find the max value in 3x3 window at current pixel's orientation
    #  0 ---------> x
    #  |
    #  |
    #  V
    #  y
    for i in range(i_h):
        for j in range(i_w):
            theta = ori[i, j]
            mag = magnitude[i, j]
            # if this is near a horizontal line, compare left and right pixels
            if theta < 22.5 or theta > 157.5:
                left = 0 if j - 1 < 0 else magnitude[i, j-1]
                right = 0 if j + 1 >= i_w else magnitude[i, j+1]
                if mag > left and mag > right:
                    output_image[i, j] = mag
            # if this is near a vertical line, compare up and down pixels
            elif theta > 67.5 and theta < 112.5:
                up = 0 if i + 1 >= i_h else magnitude[i+1, j]
                down = 0 if i - 1 < 0 else magnitude[i-1, j]
                if mag > up and mag > down:
                    output_image[i, j] = mag
            # if this is near a diagonal line
            elif theta >= 22.5 and theta <= 67.5:
                up_left = 0 if i - 1 < 0 or j - 1 < 0 else magnitude[i-1, j-1]
                down_right = 0 if i + 1 >= i_h or j + 1 >= i_w else magnitude[i+1, j+1]
                if mag > up_left and mag > down_right:
                    output_image[i, j] = mag
            # if this is near a diagonal line
            else: # theta >= 112.5 and theta <= 157.5:
                up_right = 0 if i - 1 < 0 or j + 1 >= i_w else magnitude[i-1, j+1]
                down_left = 0 if i + 1 <= i_h or j - 1 < 0 else magnitude[i+1, j-1]
                if mag > up_right and mag > down_left:
                    output_image[i, j] = mag
    
    return output_image

def iterative_BFS(img, i, j, low_threshold, high_treshold):
    '''
    Check if (i,j) connect to a strong pixel directly or indirectly via middle pixel.
    use iterative BFS
    Input:
        img:
        i, j: middle pixel, img[i, j] >= low_threshold and img[i, j] < high_threshold
        low_threshold: 
        high_threshold: 
    Return:
        A set of `seen` pixels and a boolean value, which
        If True, 
            current (i,j) and `seen` are strong pixels
        else
            current (i,j) and `seen` are weak pixels.
    '''
    i_h, i_w = img.shape
    # Use deque to store `middle` pixels we need to check
    # Only connected `middle` pixels are stored in deque
    q = deque()
    q.append((i, j))
    # Remember seen `middle`` pixels, avoid check it again
    seen = set()
    seen.add((i, j))
    # Iterative `cicle` around a pixel in 8 ways, from inner circle to outer circle.
    while len(q) > 0:
        i, j = q.popleft()
        # 4 directions to go: down -> right -> up -> left (0,1)(1,0)(0,-1)(-1,0)
        offset1 = [0, 1, 0, -1, 0]
        # Another 4 directions to go: downright -> downright -> downleft -> upleft (1,1)(1,-1)(-1,-1)(-1,1)
        offset2 = [1, 1, -1, -1, 1]
        # check connectoin in 8 ways
        for k in range(4):
            ti = i + offset1[k]
            tj = j + offset1[k+1]
            if ti >= 0 and ti < i_h and tj >= 0 and tj < i_w:
                if (ti, tj) in seen:
                    pass
                # when a pixel is strong, all seen pixels are strong
                elif img[ti, tj] >= high_treshold:
                    return seen, True
                elif img[ti, tj] >= low_threshold:
                    seen.add((ti, tj))
                    q.append((ti, tj))
            ti = i + offset2[k]
            tj = j + offset2[k+1]
            if ti >= 0 and ti < i_h and tj >= 0 and tj < i_w:
                if (ti, tj) in seen:
                    pass
                elif img[ti, tj] >= high_treshold:
                    return seen, True
                elif img[ti, tj] >= low_threshold:
                    seen.add((ti, tj))
                    q.append((ti, tj))

    # no strong pixel found, all seen pixels are weak
    return seen, False

                    
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
    # Link to strong set
    link_to_strong = set()
    # Not strong set
    not_strong = set()
    for i in range(i_h):
        for j in range(i_w):
            # If pixel is strong, keep it
            if img[i, j] >= high_threshold:
                output_image[i, j] = 255
            # If pixel is weak, check if it is linked to a strong pixel
            elif img[i, j] >= low_threshold:
                if (i, j) in link_to_strong:
                    output_image[i, j] = 255
                elif (i, j) in not_strong:
                    pass
                else:
                    seen_pixels, is_strong = iterative_BFS(img, i, j, low_threshold, high_threshold)
                    if is_strong:
                        output_image[i, j] = 255
                        link_to_strong.update(seen_pixels)
                    else:
                        not_strong.update(seen_pixels)
    return output_image


def main():
    # Initialize values
    # You can choose any sigma values like 1, 0.5, 1.5, etc
    sigma = 0.5
    size = 11

    # 1. Read the image in grayscale mode using opencv
    I = cv2.imread(r'C:\Users\a\Downloads\CAP5415\22090.jpg', cv2.IMREAD_GRAYSCALE)

    # 2. Create a one-dimensional gaussian kernel. Returns 1XN matrix
    G = gaussian_kernel(size=size, sigma=sigma)


    # Convolution of G and I
    I_xx = convolution_1d(I, G)
    I_yy = convolution_1d(I, G.T)


    # 3. First Derivative of Gaussian
    G_x = gaussian_first_derivative_kernel(size=size, sigma=sigma)

    # 4. Convolve I with G_x in x and y direction
    I_x_prime = convolution_1d(I, G_x)
    I_y_prime = convolution_1d(I, G_x.T)    

    # 5. Compute magnitude
    Mag = np.sqrt(np.square(I_x_prime) + np.square(I_y_prime))

    # Compute orientation
    # np.arctan2() returns radian, in the range [-pi, pi]. pi radians = 180 degrees.
    Ori = np.arctan2(I_y_prime, I_x_prime) 

    # 6. Compute non-max suppression
    M_nms = non_max_suppression(Mag, Ori)

    # convert to uint8 for display
    cv_Mag = (Mag / np.max(Mag) * 255).astype(np.uint8)
    cv_M_nms = (M_nms / np.max(M_nms) * 255).astype(np.uint8)

    window = cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
    # cv2.imshow("Images", I)
    cv2.imshow("Images", np.hstack((cv_Mag, cv_M_nms)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # 7. Hysteresis thresholding
    M_thresholded = hysteresis_thresholding(M_nms, 0.1, 0.2)

    # use opencv's canny to compare the result
    edge = cv2.Canny(I, 100, 200)

    plt.subplot(231),plt.imshow(I_xx, cmap = 'gray')
    plt.title('Gaussian along x Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(232),plt.imshow(I_yy,cmap = 'gray')
    plt.title('Gaussian along y Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(233),plt.imshow(I_x_prime, cmap = 'gray')
    plt.title('I_x_prime'), plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.imshow(I_y_prime, cmap = 'gray')
    plt.title('I_y_prime'), plt.xticks([]), plt.yticks([])
    plt.subplot(235),plt.imshow(Mag, cmap = 'gray')
    plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
    plt.subplot(236),plt.imshow(M_thresholded, cmap = 'gray')
    plt.title('my canny'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

    
if __name__ == '__main__':
    main()
    