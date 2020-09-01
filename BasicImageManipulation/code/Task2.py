import numpy as np
import matplotlib.pyplot as plt
import cv2


#Part 2.1
img_path = 'C:\ComputerVision\Report1\Grayscale-Lena-Image.png'
gray = cv2.imread(img_path, 0)
#0 to read it in grayscale mode, 1 for color, -1 for unchanged
complement = 255 - gray

fig, ax = plt.subplots(1, 2)
fig.suptitle('Complement of grayscale image')
ax[0].imshow(gray, cmap = plt.cm.gray)
ax[0].set(title='The original grayscale img')

ax[1].imshow(complement, cmap = plt.cm.gray)
ax[1].set(title='The complement grayscale img')
plt.show()

#Part 2.2
flipped = np.zeros(gray.shape)
flipped[:,:] = gray[::-1,:]

fig, ax = plt.subplots(1, 2)
fig.suptitle('Inverted of grayscale image')
ax[0].imshow(gray, cmap = plt.cm.gray)
ax[0].set(title='The original grayscale img')

ax[1].imshow(flipped, cmap = plt.cm.gray)
ax[1].set(title='The flipped grayscale img')
plt.show()

#Part 2.3
color = cv2.imread('C:\ComputerVision\Report1\RGBI.jpg', 1)
color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


red = color[:,:,0].copy()
blue = color[:,:,2].copy()


color_new = color.copy()

color_new[:,:,0] = blue
color_new[:,:,2] = red
color_new[:,:,1] = color[:,:,1]

fig, ax = plt.subplots(1, 2)
fig.suptitle('Red and Blue Swapping of color image')
ax[0].imshow(color)
ax[0].set(title='The original img')

ax[1].imshow(color_new)
ax[1].set(title='Img after swapping r,b channels')
plt.show()

#Part 2.4
average = np.uint8((gray + flipped)/2)

fig, ax = plt.subplots(1, 3)
fig.suptitle('Averaging of original and flipped image')
ax[0].imshow(gray, cmap=plt.cm.gray)
ax[0].set(title='The original img')

ax[1].imshow(flipped, cmap=plt.cm.gray)
ax[1].set(title='The flipped image')

ax[2].imshow(average, cmap=plt.cm.gray)
ax[2].set(title='The average image')
plt.show()

#Part 2.5
added_img = 0
added_img = gray + np.random.randint(0,255)

added_img[added_img<0] = 0 #clipping values less than 0
added_img[added_img>255] = 255 #clipping values higher than 255

fig, ax = plt.subplots(1, 2)
fig.suptitle('Add Random value, then clip')
ax[0].imshow(gray, cmap = plt.cm.gray)
ax[0].set(title="The original grayscale img")

ax[1].imshow(added_img, cmap = plt.cm.gray)
ax[1].set(title="The added+clipped img")
plt.show()