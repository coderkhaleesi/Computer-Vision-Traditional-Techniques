import numpy as np
import math
import matplotlib.pyplot as plt
import cv2


# input arguments: point to rotate, origin and angle of rotation
def rotate(point, origin, theta):

    radians = (theta*np.pi)/180
    x, y = point
    o1, o2 = origin

    final_x = o1 + math.cos(radians) * (x - o1) + math.sin(radians) * (y - o2)
    final_y = o2 + -math.sin(radians) * (x - o1) + math.cos(radians) * (y - o2)

    return final_x, final_y


def rotate_forward_map(img, theta):
    #since it's a square image
    dst = np.zeros(img.shape)
    h, w = img.shape

    center = (h/2,w/2)

    for x in range(h):
        for y in range(w):
            u, v = rotate((x,y), center, theta)
            u = int(u)
            v = int(v) #no interpolation technique applied here, just converted to int

            if (u >= 0 and u < h) and (v >=0 and v < w):
                dst[u, v] = img[x,y]

    return dst

I = cv2.imread('face 02 7043565.jpg', 1);
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
I = cv2.resize(I, (512, 512), interpolation = cv2.INTER_AREA)

rotated_0 = rotate_forward_map(I, 0)
rotated_1 = rotate_forward_map(I, -90)
rotated_2 = rotate_forward_map(I, -45)
rotated_3 = rotate_forward_map(I, -15)
rotated_4 = rotate_forward_map(I, 45)
rotated_5 = rotate_forward_map(I, 90)

fig, ax = plt.subplots(3, 2)

ax[0,0].imshow(rotated_0, cmap=plt.cm.gray)
ax[0,0].set(title="Original Image")

ax[0,1].imshow(rotated_1, cmap=plt.cm.gray)
ax[0,1].set(title="Rotated by -90")

ax[1,0].imshow(rotated_2, cmap=plt.cm.gray)
ax[1,0].set(title="Rotated by -45")

ax[1,1].imshow(rotated_3, cmap=plt.cm.gray)
ax[1,1].set(title="Rotated by -15")

ax[2,0].imshow(rotated_4, cmap=plt.cm.gray)
ax[2,0].set(title="Rotated by 45")

ax[2,1].imshow(rotated_5, cmap=plt.cm.gray)
ax[2,1].set(title="Rotated by 90")

plt.tight_layout()
plt.show()


def backward_map(img, theta):
    dst = np.zeros(img.shape)
    h, w = img.shape

    center = (h/2,w/2)

    #we iterate over destination points rather than original image points
    for u in range(h):
        for v in range(w):
            x, y = rotate((u,v), center, -theta) #the negative is added because we are doing reverse mapping
            x = int(x)
            y = int(y)

            if (x > 0 and x < h) and (u > 0 and y < w):
                dst[u, v] = img[x, y]

    return dst

backward_rotated_1 = backward_map(I, -45)
backward_rotated_2 = backward_map(I, -15)
backward_rotated_3 = backward_map(I, 45)

fig, ax = plt.subplots(3, 2)

ax[0,0].imshow(rotated_2, cmap=plt.cm.gray)
ax[0,0].set(title="Forward Map Rotate -45")

ax[0,1].imshow(backward_rotated_1, cmap=plt.cm.gray)
ax[0,1].set(title="Backward Map Rotate -45")

ax[1,0].imshow(rotated_3, cmap=plt.cm.gray)
ax[1,0].set(title="Forward Map Rotate -15")

ax[1,1].imshow(backward_rotated_2, cmap=plt.cm.gray)
ax[1,1].set(title="Backward Map Rotate -15")

ax[2,0].imshow(rotated_4, cmap=plt.cm.gray)
ax[2,0].set(title="Forward Map Rotate 45")

ax[2,1].imshow(backward_rotated_3, cmap=plt.cm.gray)
ax[2,1].set(title="Backward Map Rotate 45")

plt.tight_layout()
plt.show()

def nn_interpolation(orig_x, orig_y):
    #for nearest neighbour interpolation, we find distance from the surrounding points and get the nearest one

    x = int(np.round(orig_x))
    y = int(np.round(orig_y))

    neighbours = [[x, y], [x+1, y], [x, y+1], [x+1, y+1]]

    distances = [np.linalg.norm(np.array(n) - np.array([x,y])) for n in neighbours]

    index = np.argmin(distances)
    return (neighbours[index][0], neighbours[index][1])

def forward_map_nn_interpolation(img, theta):
    #since it's a square image
    dst = np.zeros(img.shape)
    h, w = img.shape

    center = (h/2,w/2)

    for x in range(h):
        for y in range(w):
            u, v = rotate((x,y), center, theta)
            u, v = nn_interpolation(u, v)

            if (u >= 0 and u < h) and (v >=0 and v < w):
                dst[u, v] = img[x,y]

    return dst

plt.imshow(forward_map_nn_interpolation(I, 45), cmap=plt.cm.gray)
plt.show()

def bilinear_interpolation(Img, orig_x, orig_y):

    x = int(orig_x)
    y = int(orig_y)
    c = 0

    x1 = x
    x2 = x+1
    y1 = y
    y2 = y+1

    if (x1 < Img.shape[0] and x2 < Img.shape[0]) and (y1 < Img.shape[1] and y2 < Img.shape[1]):
        a = ((x2-orig_x)/(x2-x1))*Img[x1][y1] + ((orig_x-x1)/(x2-x1))*Img[x2][y1]
        b = ((x2-orig_x)/(x2-x1))*Img[x1][y2] + ((orig_x-x1)/(x2-x1))*Img[x2][y2]

        c = ((y2-orig_y)/(y2-y1))*a + ((orig_y-y1)/(y2-y1))*b
    return c

def forward_map_bilinear_interpolation(img, theta):

    dst = np.zeros(img.shape)
    h, w = img.shape

    center = (h/2,w/2)

    for x in range(h):
        for y in range(w):
            u, v = rotate((x,y), center, -theta)
            value = bilinear_interpolation(img, u, v)

            if (x >= 0 and x < h) and (y >=0 and y < w):
                dst[x][y] = value

    return dst


plt.imshow(forward_map_bilinear_interpolation(I, 45), cmap=plt.cm.gray)
plt.show()