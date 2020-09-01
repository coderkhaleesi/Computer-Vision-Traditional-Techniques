import numpy as np
import matplotlib.pyplot as plt
import cv2
from os import listdir
from matplotlib import image
from matplotlib.pyplot import imread

train_path = 'Yale-FaceA/trainingset/'
test_path = 'Yale-FaceA/testset/'

###########################Part 1: Loading and creating data matrix ##############################
##################################################################################################

loaded_images = []
for filename in listdir(train_path):
    im = cv2.imread(train_path+filename, 0) #grayscale images
    im = im.flatten('F')                    #flatten the images to a 45045x1 vector for each img
    loaded_images.append(im)


all_data = np.stack(loaded_images).T        #stack the loaded data to form a matrix

#############################Part 2: Preprocess the data #########################################
##################################################################################################

def preprocess(A_pp):

    means = np.zeros((A_pp.shape[0],1))

    for i in range(A_pp.shape[0]):
        means[i] = np.mean(A_pp[i,:], axis=0)  # calculation of the mean of the data

    means_stack = np.tile(means, (1,A_pp.shape[1]))

    Q = A_pp - means_stack                     #centering the data
    A = None
    Q_norms = np.zeros((A_pp.shape[0],1))

    for i in range(A_pp.shape[0]):            #In this code block, the data is normalized by
        x = np.abs(np.amax(Q[i,:]))           # the infinity norm to stabilize the results better
        if x==0:
            Q_norms[i] = 1
        else:
            Q_norms[i] = x


    A = Q/Q_norms                           #data divided by L_infinity norm
    A_means = means
    return A, Q_norms, A_means              #centered data, infinity norm and means returned

A, Q_norms, A_means = preprocess(all_data)

#########################Part 3: Calculate eigenvectors and coefficients #########################
##################################################################################################

#Display the mean face
R = A_means.reshape((231,195),  order='F')
R = np.real(R)
plt.imshow(R, cmap=plt.cm.gray)
plt.show()

def eigen_faces(A_pp):
    A, Q_norms, A_means = preprocess(A_pp) #returns centered data and mu

    w,v = np.linalg.eigh(A.T@A)             #eigendecomposition of A.T@A
    F = A@v                                 #calculate the eigenvectors of the original data cov matrix (A@A.T)

    F = F/np.linalg.norm(F, axis = 0)       #normalize the eigenvectors, these are our eigenfaces
    C = A.T@F                               #coefficients of each image in terms of eigenvector basis

    D = np.diag(w, k=0)                     #eigenvalues

    return C, F, D, Q_norms, A_means

coeff, F, D, Q_norms, A_means = eigen_faces(all_data)

#The above code of preprocess and eigenface is taken from an
#Assignment I did in Intro to ML course and it's my own code.

print(f"Eigenvalues: {D}")
#so the biggest eigenvector is in 134th index

variance_retained = D[:, -15:].sum()/D.sum()
print(f"Variance retained with top 15 eigenvectors: {variance_retained}")


#############################Part 4: Display Eigenfaces #########################################
#################################################################################################

#Here I display only 4 because I reused this code to display and snip all 15 eigenfaces for my report

fig, ax = plt.subplots(2, 2)
plt.rcParams["figure.figsize"] = [15,9]
fig.suptitle('Display top 15 eigenfaces')

ax[0,0].imshow(np.real(F[:,130].reshape((231,195),  order='F')), cmap=plt.cm.gray)
ax[0,0].set(title='1st Eigenface')


ax[0,1].imshow(np.real(F[:,129].reshape((231,195),  order='F')), cmap=plt.cm.gray)
ax[0,1].set(title='2nd Eigenface')


ax[1,0].imshow(np.real(F[:,132].reshape((231,195),  order='F')), cmap=plt.cm.gray)
ax[1,0].set(title='3rd Eigenface')


ax[1,1].imshow(np.real(F[:,131].reshape((231,195),  order='F')), cmap=plt.cm.gray)
ax[1,1].set(title='4th Eigenface')
plt.show()


#####################Part 5: Build an Img recognition system and Test ############################
##################################################################################################


#Read in an image from the test dataset

Img = cv2.imread(test_path+'subject.new.01.jpg', 0)
Img = Img.flatten('F')
Img = Img[:, np.newaxis]

#center the image
Img_centered = Img - A_means

#Take first 15 principal eigenvectors
F_k = F[:, -15:]
F_k.shape

#Get the test image projection on the eigenvectors to get the 15 coefficients
Img_projection = Img_centered.T@F_k

#Get coefficients of 15 eigenvectors of all images in the dataset
coeff = C = A.T@F_k

#We now calculate nearest neighbours based on the vectors of coefficients
diff = coeff - Img_projection
norms = np.linalg.norm(diff, axis=1)

#Take the lowest 3 norms (best 3 matches) as they are least distant
indices = norms.argsort()[:3]

###########################Part 6: Display Results Of Recognition ##################################
####################################################################################################

fig, ax = plt.subplots(2, 2)
plt.rcParams["figure.figsize"] = [15,9]
fig.suptitle('Display test images and closest images')

ax[0,0].imshow(Img.reshape((231,195),  order='F'), cmap=plt.cm.gray)
ax[0,0].set(title='Original Test Img')


ax[0,1].imshow(all_data[:, indices[0]].reshape((231,195),  order='F'), cmap=plt.cm.gray)
ax[0,1].set(title='1st Match')


ax[1,0].imshow(all_data[:, indices[1]].reshape((231,195),  order='F'), cmap=plt.cm.gray)
ax[1,0].set(title='2nd Match')


ax[1,1].imshow(all_data[:, indices[2]].reshape((231,195),  order='F'), cmap=plt.cm.gray)
ax[1,1].set(title='3rd Match')
plt.show()

###########################Part 7: Processing of my images #########################################
######################### Commented because no need to run now #######################################

# for i in range(0,10):
#     fname = f"subject.tanya.{i}.jpg"
#     img = cv2.imread(fname, 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(img, (195, 231), interpolation = cv2.INTER_AREA)
#     fname = f"subject.new.{i}.jpg"
#     cv2.imwrite(fname, img)