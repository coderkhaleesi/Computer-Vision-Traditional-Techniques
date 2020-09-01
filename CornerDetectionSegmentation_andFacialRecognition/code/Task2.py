import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

filename = 'mandm.png'

#Read in the file and conver to Lab format
im = cv2.imread(filename)
im = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)


fig, ax = plt.subplots(1, 1)
fig.suptitle('Display original img in Lab format')
ax.imshow(im)
plt.show()

#############################Part 1: Create data matrix##########################################
#################################################################################################

num_features=5 #5 #Do we want to take only L, a, b or L, a, b, x, and y
data_matrix = np.zeros((im.shape[0]*im.shape[1], num_features))
t=0
#create the data matrix by filling in L, a, b, x, and y values. It creates a matrix with rows=widthxheight
#and columns=5 or columns=3 matrix
for u in range(im.shape[0]):
    for v in range(im.shape[1]):
        data_matrix[t][0] = im[u][v][0]
        data_matrix[t][1] = im[u][v][1]
        data_matrix[t][2] = im[u][v][2]
        data_matrix[t][3] = u
        data_matrix[t][4] = v
        t+=1

#############################Part 2: Init Centroids (both normal and Kmeans++)###################
#################################################################################################

#init centroids randomly from points in the data matrix
def init_centroids(data, k):
    c = np.zeros((k, data.shape[1]))
    indices = random.sample(range(0, data.shape[0]), k)
    c = data[indices]
    return c

#distance square for Kmeans plus plus
def distance(p1, p2):
    return np.sum((p1 - p2)**2)


def init_centroids_plus_plus(data, k):

    #centroid list, we keep adding to it as we find more centroids till we reach k
    c = []
    #add the first centroid randomly from the data
    c.append(data[np.random.randint(data.shape[0]), :])
    print(c)

    #loop over rest of the k and add centroids one by one
    for c_id in range(0,k-1):
        dist = []
        for i in range(data.shape[0]): #for each data point, we find distance to closest centroid
            point = data[i, :]
            d = sys.maxsize
            for j in range(len(c)):
                temp_dist = distance(point, c[j]) #only take centroids already in c
                d = min(d, temp_dist)    #take min distance as we want closest centroid
            dist.append(d)
        dist = np.array(dist)
        next_c = data[np.argmax(dist), :]  #find the point which has highest distance from its closest centroid
        c.append(next_c)
        dist = []

    return c

#Reference https://www.geeksforgeeks.org/ml-k-means-algorithm/


#############################Part 3: E and M step code###########################################
#################################################################################################

def e_step(c, data):
    belong_vectors = np.zeros((data.shape)) #initialize a matrix that tells us which point belongs to which cluster
    for i in range(data.shape[0]):               #iterate over all the points
        min_dist = 999999999999                  #initialize unusually high min distance for each point/iteration
        for j in range(c.shape[0]):              #iterate over all the centroids
            if min_dist > np.linalg.norm(data[i]-c[j]):
                belong_vectors[i] = c[j]         #now fill this matrix with the clutser closest to each point
                min_dist = np.linalg.norm(data[i]-c[j]) #update min distance for that point

    #print(belong_vectors)
    return belong_vectors

def m_step(belong_vectors, data, c):
    new_c = np.zeros(c.shape)          #In order to update centroids in m step, we need an empty matrix
    for i in range(c.shape[0]):        #iterate over centroids
        sum_points = np.zeros((1, c.shape[1]))
        count_points = 0
        for j in range(data.shape[0]):            #iterate over all points in data matrix
            if (belong_vectors[j] == c[i]).all(): #if the data point belongs to the cluster
                sum_points+=data[j]               #add it to the sum
                #print(sum_points.shape)
                count_points+=1                   #for keeping count of points in a cluster

        new_c[i] = sum_points/count_points       #take the average of all points in that cluster
                                            #and assign result to the centroid (that's the definition of a centroid)
    return new_c                       #return new centroids


#############################Part 4: Execute K-Means ############################################
#################################################################################################

def my_kmeans(data, num_centroids, iterations): #finally the kmeans algorithm
    #c = init_centroids(data, num_centroids)     #init centroids
    c = init_centroids_plus_plus(data, num_centroids)
    c = np.asarray(c)
    last_c = c                                 #assign the last iteration centroids to c for now

    for i in range(iterations):
        belong_vectors = e_step(c, data)     #E step
        c = m_step(belong_vectors, data, c)  #M step
        if (c == last_c).all():              #if new centroids are equal to last iteration centroids, break the loop
            break
        last_c = c                           #reassign

    return belong_vectors, c

#data, num_centroids, iterations as arguments
b, c = my_kmeans(data_matrix, 12, 30)


#########################Part 5: Create & display segmented image################################
#################################################################################################
final_image = np.zeros((im.shape[0], im.shape[1], 3))
t=0
#reconstruct the segmented image in a way that each point gets assigned to its cluster
for u in range(im.shape[0]):
    for v in range(im.shape[1]):
        final_image[u][v][0] = b[t][0]  #we use t because our b vector was flattened in the same way
        final_image[u][v][1] = b[t][1]  #and now we fill our image using the centroid values for each point
        final_image[u][v][2] = b[t][2]
        t+=1

final_image = np.uint8(final_image)

fig, ax = plt.subplots(1, 1)
fig.suptitle('Display Segmented img in Lab format')

ax.imshow(final_image)
plt.show()
