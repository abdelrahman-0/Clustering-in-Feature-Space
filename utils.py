import numpy as np
import pandas as pd
import torchvision.models as models
from torchvision import transforms
import cv2
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Read image and return it as torch Tensor
def get_img(name):
    
    # Read image and scale values to lie in [0,1]
    img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255
    
    # Normalize image with respect to ImageNet dataset
    T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]) 
    tensor = T(img)
    return tensor, img.shape[0], img.shape[1]

# Normalize each row in dataset to have unit length
def normalize_data(X):
    h, w, depth = X.shape
    mat = preprocessing.normalize(X.reshape((h*w, depth)), axis=1).reshape((h, w, depth))
    return mat

# Apply PCA to a dataset
def perform_PCA(X, num_pc=512, var=99.0):
    print("PCA started...")
    
    # Standarize data to 0 mean and unit variance and use it to fit a PCA model
    scaled = preprocessing.scale(X)
    pca = PCA(n_components=num_pc)
    pca_data = pca.fit_transform(scaled)
    per_var = pca.explained_variance_ratio_ * 100
    
    # Accumulate principal components until enough variance in represented in the data
    accumulated = []    
    total_variance = 0.0
    for i in range(num_pc):
        total_variance += per_var[i]
        accumulated.append(total_variance)
        if total_variance >= var:
            break
    num_pc = i+1
    accumulated = np.array(accumulated)
    print('First {0} principal components capture {1}% of the data\'s variance'.format(num_pc, round(total_variance, 2)))
    
    print("PCA ended")
    return pca_data[:, :num_pc], num_pc, accumulated[:num_pc], pca

# Nearest neighbor upsampling of 2D matrix
def upsample(X, new_y, new_x):
    old_y, old_x = X.shape[:2]
    result = np.zeros((new_y, new_x), dtype=type(X[0][0]))
    for i in range(new_y):
        for j in range(new_x):
            result[i][j] = X[int(i/new_y*old_y)][int(j/new_x*old_x)]
    return result

# Cluster n-by-d data and return the cluster indices of each datapoint
def cluster_all(X, n_clusters=10):
    fitted = KMeans(n_clusters=n_clusters, init='random', n_init=100).fit(X)
    labels = fitted.labels_
    return labels