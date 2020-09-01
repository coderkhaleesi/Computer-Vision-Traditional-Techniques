# -*- coding: utf-8 -*-
# CLAB3
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2

I = Image.open('stereo2012a.jpg');
plt.imshow(I)
uv = plt.ginput(12) # Graphical user interface to get 6 points

#xyz coordinates (world coordinates)
xyz = np.asarray([[7, 7, 0],
                    [14, 7, 0],
                    [21, 7, 0],
                    [21, 14, 0],
                    [0, 7, 7],
                    [0, 14, 7],
                    [0, 21, 7],
                    [0, 21, 14],
                    [7, 0, 7],
                    [7, 0, 14],
                    [7, 0, 21],
                    [14, 0, 21]])

'''
%% TASK 1: CALIBRATE
%
% Function to perform camera calibration
%
% Usage:   calibrate(image, XYZ, uv)
%          return C
%   Where:   image - is the image of the calibration target.
%            XYZ - is a N x 3 array of  XYZ coordinates
%                  of the calibration target points.
%            uv  - is a N x 2 array of the image coordinates
%                  of the calibration target points.
%            K   - is the 3 x 4 camera calibration matrix.
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of the calibration
%  target.
%
%  It also projects the XYZ coordinates back into image coordinates using
%  the calibration matrix and plots these points too as
%  a visual check on the accuracy of the calibration process.
%
%  Lines from the origin to the vanishing points in the X, Y and Z
%  directions are overlaid on the image.
%
%  The mean squared error between the positions of the uv coordinates
%  and the projected XYZ coordinates is also reported.
%
%  The function should also report the error in satisfying the
%  camera calibration matrix constraints.
%
'''


#####################################################################
def calibrate(im, XYZ, uv):
    X = XYZ[:, 0] #get X, Y, and Z
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]

    u = [x[0] for x in uv] #Get u and v separately from tuples
    v = [x[1] for x in uv]

    num_points = XYZ.shape[0] #get the number of points marked

    #declare matrices A and b
    A = np.zeros((num_points*2, 11))
    b = np.zeros((num_points*2, 1))

    j=0
    for i in range(0, num_points*2, 2):
        #DLT algorithm from lectures
        A[i] = [X[j], Y[j], Z[j], 1, 0, 0, 0, 0, -u[j]*X[j], -u[j]*Y[j], -u[j]*Z[j]]
        A[i+1] = [0, 0, 0, 0, X[j], Y[j], Z[j], 1, -v[j]*X[j], -v[j]*Y[j], -v[j]*Z[j]]
        b[i] = u[j]
        b[i+1] = v[j]
        j += 1

    #The matrix is the solution to a linear least squares problem ||Ax - b||^2
    C = np.linalg.lstsq(A, b, rcond=None)[0]

    #these two should be equal, verify by printing
    # print(A@C)
    # print(uv)

    newrow = [1]
    C = np.vstack([C, newrow]) #append 1 (last entry) so that it can be reshaped to 3x4
    C = C.reshape((3,4))
    print(f"{C}")
    return C

#This function is used to reconstruct u, v from the calibration matrix and X,Y,Z coordinates
def reconstruct_from_C(C, XYZ):
    num_points = XYZ.shape[0]
    XYZ = XYZ.T
    newrow = np.repeat([1], num_points)
    XYZ = np.vstack([XYZ, newrow]) #convert to homogenous coordinates

    reconstructed_uv = np.zeros((num_points, 2))
    uvw = C@XYZ

    r_u = uvw[0, :]/uvw[2, :]
    reconstructed_uv[:, 0] = r_u.T
    r_v = uvw[1, :]/uvw[2, :]
    reconstructed_uv[:, 1] = r_v.T #convert back to cartesian

    return reconstructed_uv

C = calibrate(I, xyz, uv)
recons_uv = reconstruct_from_C(C, xyz)

error = ((recons_uv - uv)**2).mean()
print(f"The error between actual u,v and reconstructed u,v from calibration matrix is: {error}")

u1 = [x[0] for x in uv]
v1 = [x[1] for x in uv]

u2 = recons_uv[:, 0]
v2 = recons_uv[:, 1]

plt.show()
plt.imshow(I)
plt.scatter(u1, v1, c='red',  s=5, marker='x')
plt.scatter(u2, v2, c='blue',  s=4, marker='x')
plt.show()




############################################################################

'''
%% TASK 2:
% Computes the homography H applying the Direct Linear Transformation
% The transformation is such that
% p = np.matmul(H, p.T), i.e.,
% (uBase, vBase, 1).T = np.matmul(H, (u2Trans , v2Trans, 1).T)
% Note: we assume (a, b, c) => np.concatenate((a, b, c), axis), be careful when
% deal the value of axis
%
% INPUTS:
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p')
% uBase, vBase - vectors with coordinates u and v of the original base image point p
%
% OUTPUT
% H - a 3x3 Homography matrix
%
% your name, date
'''


I_left = Image.open('left.jpg');
plt.imshow(I_left)
try:
    uv_circ = plt.ginput(6) # Graphical user interface to get 6 points
except Exception as e:
    print("ginput 1 failed")
plt.close('all')
u_circ = [x[0] for x in uv_circ]
v_circ = [x[1] for x in uv_circ]

I_right = Image.open('right.jpg');
plt.imshow(I_right)
try:
    uv = plt.ginput(6) # Graphical user interface to get 6 points
except Exception as e:
    print("ginput 2 failed")

plt.close('all')
u_base = [x[0] for x in uv]
v_base = [x[1] for x in uv]

def homography(u2Trans, v2Trans, uBase, vBase):

    num_points = len(u2Trans)
    A = np.zeros((num_points*2, 9))

    j=0
    for i in range(0, num_points*2, 2): #Mapping points using formula from lectures
        print(i)
        A[i] = [u2Trans[j], v2Trans[j], 1, 0, 0, 0, -u2Trans[j]*uBase[j], -uBase[j]*v2Trans[j], -uBase[j]]
        A[i+1] = [0, 0, 0, u2Trans[j], v2Trans[j], 1, -u2Trans[j]*vBase[j], -v2Trans[j]*vBase[j], -vBase[j]]
        j += 1

    u, s, vh = np.linalg.svd(A, full_matrices=True) #SVD to solve the linear equation

    H = vh[-1, :]/vh[-1,-1]
    return H

H_matrix = homography(u_circ, v_circ, u_base, v_base)

#This function 'test_homography' uses the homography matrix calculated above to reconstruct points and compare original and
#reconstructed points

def test_homography(H, base, circ):

    newrow = np.repeat([1],circ.shape[1])
    circ = np.vstack([circ, newrow])

    #H = H/H[-1]
    print(H)
    x = H.reshape((3,3))@circ

    r_u = x[0, :]/x[2, :]
    r_v = x[1, :]/x[2, :]
    reconstructed_base = np.asarray([r_u, r_v])
    print(reconstructed_base)
    print(base)

circ = np.asarray([u_circ, v_circ])
print(circ.shape)
base = np.asarray([u_base, v_base])
test_homography(H_matrix, base, circ)

#Function to warp left image
def warp_img(img, H):
    #since it's a square image
    dst = np.zeros(img.shape)
    h, w = img.shape
    # print(h)
    # print(w)

    for x in range(h):
        for y in range(w):
            newrow = np.repeat([1],1)
            init_coords = np.vstack([x, y, newrow])
            u, v, s = H.reshape((3,3))@init_coords
            u = int(u/s)
            v = int(v/s) #no interpolation technique applied here, just converted to int

            if (u >= 0 and u < h) and (v >=0 and v < w):
                dst[u, v] = img[x,y]
    return dst

I = cv2.imread('Left.jpg', 1);
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
warped = warp_img(I, H_matrix)
plt.imshow(warped, cmap=plt.cm.gray)
plt.show()


############################################################################
def rq(A):
    # RQ factorisation

    [q,r] = np.linalg.qr(A.T)   # numpy has QR decomposition, here we can do it
                                # with Q: orthonormal and R: upper triangle. Apply QR
                                # for the A-transpose, then A = (qr).T = r.T@q.T = RQ
    R = r.T
    Q = q.T
    return R,Q

