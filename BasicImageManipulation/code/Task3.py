import numpy as np
import matplotlib.pyplot as plt
import cv2

#please comment out this code before running. It will fail because 1.jpg, 2.jpg, 3.jpg are not there
#because you didn't ask us to submit those

#Reading and resizing face images
one = cv2.imread('1.jpg', 1);
onef = cv2.resize(one, (1024, 720));
cv2.imwrite('face 01 7043565.jpg', onef, [int(cv2.IMWRITE_JPEG_QUALITY), 90]);

two = cv2.imread('2.jpg', 1);
twof = cv2.resize(two, (1024, 720));
cv2.imwrite('face 02 7043565.jpg', twof, [int(cv2.IMWRITE_JPEG_QUALITY), 90]);

three = cv2.imread('3.jpg', 1);
threef = cv2.resize(three, (1024, 720));
cv2.imwrite('face 03 7043565.jpg', threef, [int(cv2.IMWRITE_JPEG_QUALITY), 90]);

#Part 3.a
I = cv2.imread('face 01 7043565.jpg', 1) #opencv reads in BGR
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB) #convert BGR to RGB
Img = cv2.resize(I, (768, 512), interpolation = cv2.INTER_AREA); #coz matplotlib displays in RGB

#Part 3.b
rch = Img[:,:,0]
gch = Img[:,:,1]
bch = Img[:,:,2]


fig, ax = plt.subplots(2, 2)
fig.suptitle('Display img and channel imgs')

ax[0,0].imshow(Img)
ax[0,0].set(title="The original img")

ax[0,1].imshow(rch, cmap=plt.cm.gray)
ax[0,1].set(title="Red Channel")

ax[1,0].imshow(gch, cmap=plt.cm.gray)
ax[1,0].set(title="Green Channel")

ax[1,1].imshow(bch, cmap=plt.cm.gray)
ax[1,1].set(title="Blue Channel")

plt.show()

#Part 3.c
histr = cv2.calcHist([rch], [0], None, [256], [0,256])
histg = cv2.calcHist([gch], [0], None, [256], [0,256])
histb = cv2.calcHist([bch], [0], None, [256], [0,256])

plt.title('Histograms of RGB channels')
plt.plot(histr, color='r')
plt.plot(histg, color='g')
plt.plot(histb, color='b')
#showing in the same graph for comparison

plt.show()

#Part 3.d

rch_eq = cv2.equalizeHist(rch)
gch_eq = cv2.equalizeHist(gch)
bch_eq = cv2.equalizeHist(bch)

Img_eq = Img.copy()
Img_eq[:,:,0] = rch_eq
Img_eq[:,:,1] = gch_eq
Img_eq[:,:,2] = bch_eq

#Showing Histograms
histr_eq = cv2.calcHist([rch_eq], [0], None, [256], [0,256])
histg_eq = cv2.calcHist([gch_eq], [0], None, [256], [0,256])
histb_eq = cv2.calcHist([bch_eq], [0], None, [256], [0,256])

plt.title('Histograms of RGB channels after equalization')
plt.plot(histr_eq, color='r')
plt.plot(histg_eq, color='g')
plt.plot(histb_eq, color='b')
plt.show()

#Showing Images
fig, ax = plt.subplots(2, 2)
fig.suptitle('Equalized img and equalized channel imgs')

ax[0,0].imshow(Img_eq)
ax[0,0].set(title="The original img")

ax[0,1].imshow(rch_eq, cmap=plt.cm.gray)
ax[0,1].set(title="Red Channel")

ax[1,0].imshow(gch_eq, cmap=plt.cm.gray)
ax[1,0].set(title="Green Channel")

ax[1,1].imshow(bch_eq, cmap=plt.cm.gray)
ax[1,1].set(title="Blue Channel")

plt.show()