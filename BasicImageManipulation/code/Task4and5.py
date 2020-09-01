import numpy as np
import matplotlib.pyplot as plt
import cv2

#please make sure to keep the face images in the same folder as code
#before running this

#Task 4
#Part 1
I = cv2.imread('face 02 7043565.jpg', 1);
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
I = np.clip(I, 0, 255)

J = I[50:600, 200:800]
#cropped the image

J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)
cropped_resized_gray = cv2.resize(J, (256, 256), interpolation = cv2.INTER_AREA)
cropped_resized_gray = np.clip(cropped_resized_gray, 0, 255)

fig, ax = plt.subplots(1, 2)
fig.suptitle('Display img and resized gray img')

ax[0].imshow(I)
ax[0].set(title="The original img")

ax[1].imshow(cropped_resized_gray, cmap=plt.cm.gray)
ax[1].set(title="Cropped resized gray img")
plt.show()

#Part 2
noise = 15*np.random.randn(256,256)
gray_with_noise = cropped_resized_gray + noise

plt.imshow(gray_with_noise, cmap=plt.cm.gray)

#Part 3
hist_without_noise = cv2.calcHist([cropped_resized_gray], [0], None, [256], [0,256])
hist_with_noise = cv2.calcHist([np.clip(np.uint8(gray_with_noise), 0, 255)], [0], None, [256], [0,256])

fig, ax = plt.subplots(1, 2)
fig.suptitle('Added Gaussian noise')

ax[0].plot(hist_without_noise)
ax[0].set(title="Without Noise")

ax[1].plot(hist_with_noise)
ax[1].set(title="With Noise")
plt.show()

#Part 4

#First let's create a Gaussian kernel
kernel_1 = cv2.getGaussianKernel(5, 1) #size, std is 1
kernel_5 = cv2.getGaussianKernel(5, 5) #std=5
kernel_10 = cv2.getGaussianKernel(5, 10) #std=10
kernel_15 = cv2.getGaussianKernel(5, 15) #std=15
window = np.outer(kernel_1, kernel_1.T) #the 5x5 kernel

#The convolution function
def conv(img, kernel):
    img_row, img_col = img.shape
    kernel_row, kernel_col = kernel.shape
    final_img = np.zeros(img.shape)

    pad_h = int((kernel_row-1)/2)
    pad_w = int((kernel_col-1)/2)

    padded_img = np.zeros((img_row+2*pad_h, img_col+2*pad_w))

    padded_img[pad_h:padded_img.shape[0]-pad_h,pad_w:padded_img.shape[1]-pad_w] = img

    for i in range(img_row):
        for j in range(img_col):
            final_img[i, j] = np.sum(kernel*padded_img[i:i+kernel_row, j:j+kernel_col])

    return final_img

#Part 5
output_img = conv(gray_with_noise, window)

fig, ax = plt.subplots(1, 2)
fig.suptitle('Applying custom Gaussian filter')

ax[0].imshow(gray_with_noise, cmap=plt.cm.gray)
ax[0].set(title="Gray img with noise")

ax[1].imshow(output_img, cmap=plt.cm.gray)
ax[1].set(title="Gray img after filter applied")
plt.show()

fig, ax = plt.subplots(2, 2)

ax[0,0].imshow(output_img, cmap=plt.cm.gray)
ax[0,0].set(title="Gaussian Filter with std 1")

ax[0,1].imshow(conv(gray_with_noise, np.outer(kernel_5, kernel_5.T)), cmap=plt.cm.gray)
ax[0,1].set(title="Gaussian Filter with std 1")

ax[1,0].imshow(conv(gray_with_noise, np.outer(kernel_10, kernel_10.T)), cmap=plt.cm.gray)
ax[1,0].set(title="Gaussian Filter with std 10")

ax[1,1].imshow(conv(gray_with_noise, np.outer(kernel_15, kernel_15.T)), cmap=plt.cm.gray)
ax[1,1].set(title="Gaussian Filter with std 15")

plt.tight_layout()
plt.show()


#Part 6
output_img = conv(gray_with_noise, np.outer(kernel_15, kernel_15.T))
output_with_built_in = cv2.GaussianBlur(gray_with_noise, (5,5), 15)
#https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#void%20GaussianBlur(InputArray%20src,%20OutputArray%20dst,%20Size%20ksize,%20double%20sigmaX,%20double%20sigmaY,%20int%20borderType)

fig, ax = plt.subplots(1, 2)
fig.suptitle('Comparing custom filter with built-in filter')

ax[0].imshow(output_img, cmap=plt.cm.gray)
ax[0].set(title="Gray img after Custom filter")

ax[1].imshow(output_with_built_in, cmap=plt.cm.gray)
ax[1].set(title="Gray img after Built-in Filter")
plt.show()

hist_my_func = cv2.calcHist([np.uint8(output_img)], [0], None, [256], [0,256])
hist_built_in = cv2.calcHist([np.uint8(output_with_built_in)], [0], None, [256], [0,256])


fig, ax = plt.subplots(1, 2)
fig.suptitle('Compare histograms of filtered images')

ax[0].plot(hist_my_func)
ax[0].set(title="Hist custom function")

ax[1].plot(hist_built_in)
ax[1].set(title="Hist built-in function")
plt.show()

#Task 5

sobel_filter = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
output_sobel_vertical = conv(cropped_resized_gray, sobel_filter)
output_sobel_horizontal = conv(cropped_resized_gray, sobel_filter.T)#np.flip axis=0

fig, ax = plt.subplots(1, 3)
fig.suptitle('')

ax[0].imshow(cropped_resized_gray, cmap=plt.cm.gray)
ax[0].set(title="Original image")

ax[1].imshow(output_sobel_vertical, cmap=plt.cm.gray)
ax[1].set(title="After vertical sobel")

ax[2].imshow(output_sobel_horizontal, cmap=plt.cm.gray)
ax[2].set(title="After horizontal sobel")
plt.show()


#Comparison with in-built func
sobelx = cv2.Sobel(cropped_resized_gray,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(cropped_resized_gray,cv2.CV_64F,0,1,ksize=3)

fig, ax = plt.subplots(1, 3)
fig.suptitle('In-built Sobel Filter')

ax[0].imshow(cropped_resized_gray, cmap=plt.cm.gray)
ax[0].set(title="Original image")

ax[1].imshow(sobelx, cmap=plt.cm.gray)
ax[1].set(title="After vertical sobel")

ax[2].imshow(sobely, cmap=plt.cm.gray)
ax[2].set(title="After horizontal sobel")
plt.show()



#On more images
I = cv2.imread('face 03 7043565.jpg', 1);
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
I = np.clip(I, 0, 255)

J = I[50:600, 200:800]
#cropped the image

J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)
cropped_resized_gray = cv2.resize(J, (256, 256), interpolation = cv2.INTER_AREA)
cropped_resized_gray = np.clip(cropped_resized_gray, 0, 255)

output_sobel_vertical = conv(cropped_resized_gray, sobel_filter)
output_sobel_horizontal = conv(cropped_resized_gray, sobel_filter.T)#np.flip axis=0

fig, ax = plt.subplots(1, 3)
fig.suptitle('')

ax[0].imshow(cropped_resized_gray, cmap=plt.cm.gray)
ax[0].set(title="Original image")

ax[1].imshow(output_sobel_vertical, cmap=plt.cm.gray)
ax[1].set(title="After vertical sobel")

ax[2].imshow(output_sobel_horizontal, cmap=plt.cm.gray)
ax[2].set(title="After horizontal sobel")
plt.show()


sobelx = cv2.Sobel(cropped_resized_gray,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(cropped_resized_gray,cv2.CV_64F,0,1,ksize=3)

fig, ax = plt.subplots(1, 3)
fig.suptitle('In-built Sobel Filter')

ax[0].imshow(cropped_resized_gray, cmap=plt.cm.gray)
ax[0].set(title="Original image")

ax[1].imshow(sobelx, cmap=plt.cm.gray)
ax[1].set(title="After vertical sobel")

ax[2].imshow(sobely, cmap=plt.cm.gray)
ax[2].set(title="After horizontal sobel")
plt.show()
