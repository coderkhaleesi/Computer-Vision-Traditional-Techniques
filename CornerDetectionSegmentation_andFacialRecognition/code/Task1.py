"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID): Tanya Dixit (u7043565)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


filename = 'Harris_1.jpg'

def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    print(img.shape)
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2
thresh = 789213.28
k = 0.05
window = 3

# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()
import matplotlib.pyplot as plt

bw = plt.imread(filename)
bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
bw = np.array(bw * 255, dtype=int)
# computer x and y derivatives of image
print(bw.shape)
print(dx.shape)
Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

#The below line creates a gaussian blur filter in a way that it emphasizes the gradient when
#convolved. Basically we want to pass a window (g) onto Ix^2, Iy^2 and Ix*Iy
#so that we get the directions of maximal gradient in the neighbourhood of the point (as it is done)
#for every point
g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)

######################################################################
# Task: Compute the Harris Cornerness
######################################################################

det = Ix2*Iy2 - Ixy ** 2  #calculate the determinant of the M matrix
trace = Ix2 + Iy2 #calculate trace of the M matrix

R = det - k * trace**2 #cornerness R for each pixel is det - k*trace^2
                        #This tells us whether a particular pixel is a corner or not


######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################
#print(R)

#Now we have cornerness for each pixel, we apply non-max suppression to get the corners
output = set() #save corner points in a set so that repeating points are not marked twice in the image
for i in range(R.shape[0] - window): #iterate over each pixel in R
    for j in range(R.shape[1] - window):
        mat = R[i:i+window,j:j+window] #take a window of size window
        if np.max(mat) > thresh: #if the max in that window (mat) is greater than threshold
            #print(np.max(mat))
            (x, y) = np.unravel_index(np.argmax(mat, axis=None), mat.shape) #find out which x,y have the max value
            #print(mat.shape)
            #x, y = np.argmax(mat)
            output.add((x+i, y+j)) #save that x,y in the list of corners
        else:
            continue


#print(output)

#display code
plt.figure()
bw = plt.imread(filename)
#bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
plt.imshow(bw)
u, v = zip(*output)
plt.scatter(v, u, c='red',  s=0.1, marker='x')
plt.show()




img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,window,3,k)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>thresh]=[0,0,255] #
print(0.01*dst.max())

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)#, cmap=plt.cm.gray)
plt.show()
# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()