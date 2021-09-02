from random import shuffle

import numpy as np
import pandas as pd
import torchvision.models as models
import matplotlib as mpl
from my_functions import *

# Initialize model parmeters
vgg_depth = 29
vgg = models.vgg16(pretrained=True, progress=True)

# Get image
img, tensor = get_img(IMG_IDX, dataset='Dataset'+DATASET_IDX, show_img=True)

result = vgg.features[:vgg_depth](tensor.view(1, 3, tensor.shape[1], tensor.shape[2]))[0]
result = result.permute(1,2,0).detach().numpy().astype(np.float64)
height, width, depth = result.shape

result = normalize_img(result).reshape((-1, depth))
result = perform_PCA(result, var=99.0)[0]

first_pc = upsample(result[...,0].reshape((height, width)), img.shape[0], img.shape[1])
plt.imshow(first_pc, cmap='gray')
plt.axis('off')
plt.show()
plt.imsave('new_results/ds' + DATASET_IDX +'_depth{}_{}_FIRST_PC.png'.format(vgg_depth, IMG_IDX), first_pc, cmap='gray')

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
s = pd.Series(result[..., 0])
s.plot.hist(ax=ax, bins=50)
plt.xlabel('feature projection along first PC', fontsize=22)
plt.ylabel('count', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
fig.show()

SILHOUETTE_RANGE = range(2, 11)

# SILHOUETTE SCORE BASED NUM CLUSTERS DETERMINATION
fittings = []
scores = []
for NUM_CLUSTERS in tqdm(SILHOUETTE_RANGE, desc='Trying out different k\'s'):
    clustering_result = cluster_all(result, method=METHOD, n_cluster=NUM_CLUSTERS, sums=None, sizes=None)[0]
    fittings.append(clustering_result)
    scores.append(silhouette_score(result, clustering_result))

NUM_CLUSTERS = np.argmax(scores) + 2


print('Optimal number of clusters: ', NUM_CLUSTERS)
plt.plot(SILHOUETTE_RANGE, scores)
plt.title(meth)
plt.show()


indices, error, n_clusters = cluster_all(result, None, None, n_cluster=NUM_CLUSTERS, method=METHOD, agc_connectivity=True, neighbours=10)

plt.imshow(indices.reshape((height, width)), cmap=CMAP)
plt.axis('off')
plt.show()

check = input('Change Indices ? (y/n)')

l = permutations(np.unique(indices).tolist())
shuffle(l)
idx = 0
new_indices = indices.copy()
while check.lower() != 'n':
    for i, j in enumerate(l[idx]):
        new_indices[indices == i] = j
    plt.imshow(new_indices.reshape((height, width)), cmap=CMAP)
    plt.axis('off')
    plt.show()
    idx += 1
    idx %= len(l)
    check = input('Change Indices ? (y/n)')
indices = new_indices

r = indices.reshape((height, width))
upsampled_r = upsample(r, img.shape[0], img.shape[1])
plt.imsave('new_results/upsampled_{}.png'.format(IMG_IDX), upsampled_r, cmap=CMAP)