# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:05:10 2021

@author: Lenovo
"""

import cv2
import numpy as np
fixed_size = tuple((550, 550))
bins = 8
  
import cv2 as cv
def hsvHistogramFeatures(image):
    rows,cols,dd = image.shape
    # convertir l'image RGB en HSV.
    image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    
    h = image[...,0]
    s = image[...,1]
    v = image[...,2]

    # Chaque composante h,s,v sera quantifiée équitablement en 8x2x2
    # le nombre de niveau de quantification est:
    numberOfLevelsForH = 8
    numberOfLevelsForS = 2
    numberOfLevelsForV = 2

    # Trouver le maximum.
    maxValueForH = np.max(h)
    maxValueForS = np.max(s)
    maxValueForV = np.max(v)

    # Initialiser l'histogramme à des zéro de dimension 8x2x2
    hsvColorHisto = np.zeros((8,2,2))

    # Quantification de chaque composante en nombre niveaux étlablis
    quantizedValueForH = (h*numberOfLevelsForH/maxValueForH)
    quantizedValueForS = (s*numberOfLevelsForS/maxValueForS)
    quantizedValueForV = (v*numberOfLevelsForV/maxValueForV)

    # Créer un vecteur d'indexes
    index = np.zeros((rows*cols,3))
    index[:,0] = quantizedValueForH.flatten()
    index[:,1] = quantizedValueForS.flatten()
    index[:,2] = quantizedValueForV.flatten()

    # Remplir l'histogramme pour chaque composante h,s,v
    # (ex. si h=7,s=2,v=1 Alors incrémenter de 1 la matrice d'histogramme à la position 7,2,1)
    for i in range(len(index[:,0])):
        if(index[i,0]==0 or index[i,1]==0 or index[i,2]==0):
            continue
        hsvColorHisto[int(index[i,0]),int(index[i,1]),int(index[i,2])] +=1
    # normaliser l'histogramme à la somme
    hsvColorHisto = hsvColorHisto.flatten()
    #hsvColorHisto /= np.sum(hsvColorHisto)
    return hsvColorHisto.reshape(-1)

def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # calcule des couleur histogram la distribution des intensite des couleur de l'image
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize  histogram
    cv2.normalize(hist, hist)
    # return  histogram
    return hist.flatten()
from skimage.feature import greycomatrix, greycoprops
def  textureFeatures(img):
    """Basée sur l'analyse de textures par la GLCM (Gray-Level Co-Occurrence Matrix)
    Le vecteur de taille 1x4 contiendra [Contrast, Correlation, Energy, Homogeneity]
    """
    im = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    glcm = greycomatrix(im, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    Contrast=greycoprops(glcm, 'contrast')[0, 0]
    correlation=greycoprops(glcm, 'correlation')[0, 0]
    #energy=greycoprops(glcm, 'energy')[0, 0]
    homogeneity=greycoprops(glcm, 'homogeneity')[0, 0]
    #Il faut normaliser avant de retourner le vecteur descripteur
    
    features_texture=[Contrast,correlation,homogeneity]
    #features_texture/=np.sum(features_texture)
    return features_texture
def shapeFeatures(img):
    # https://www.pyimagesearch.com/2014/10/27/opencv-shape-descriptor-hu-moments-example/
    im = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    shapeFeat = cv.HuMoments(cv.moments(im)).flatten()
    return shapeFeat

def color_Moments(img):
    img = np.reshape(img,(-1,3))
    image_mean = np.mean(img,axis=0)
    image_std = np.std(img,0)
    colorFeatures = [image_mean[0],image_std[0],image_mean[1],image_std[1],image_mean[2],image_std[2]]
    return colorFeatures

import os
global_features = []
def Exraction_features(path,current_labell):
    files = [os.path.join(path,p) for p in sorted(os.listdir(path))]
    for f in files:
            
        image = cv2.imread(f)
        image = cv2.resize(image, fixed_size)
        histo=fd_histogram(image)
        #texture=textureFeatures(image)
        #shape=shapeFeatures(image)
        #coleur=color_Moments(image)
        global_feature=np.concatenate([histo],axis=None)
        global_features.append(global_feature)
path_car='2Classes'
Exraction_features(path_car,0)
import pandas as pd
dfX = pd.DataFrame(global_features) 
X=dfX.iloc[:,0:-1].values

#print(X)
########################## Standarisation #######################################
from sklearn.preprocessing import StandardScaler as SS
ssX= SS()
#X=ssX.fit_transform(X)
######################### Split train+test aleatoire #######################################

# birch clustering
# agglomerative clustering
from numpy import unique
from numpy import where
#from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot
model = AgglomerativeClustering(n_clusters=2)
# fit model and predict clusters
yhat = model.fit_predict(X)
print(yhat)
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
y_pred_R=[]
name=[]
for f in sorted(os.listdir(path_car)):
    name.append(f)

name = pd.DataFrame(name,columns=['Name']) 
yhat = pd.DataFrame(yhat,columns=['class']) 
result = pd.merge(name,yhat,left_index=True, right_index=True,)
result.to_csv (r'Kmeans4.csv',index = None, header=True)