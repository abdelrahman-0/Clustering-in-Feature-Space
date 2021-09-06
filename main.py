from utils import *
import sys

# Initialize variables
model = models.vgg16(pretrained=True, progress=True)   # Classification Model
model_depth = 29                                       # Layer

# Get image
img, img_h, img_w = get_img(sys.argv[1])

# Obtain feature tensor from model
result = model.features[:model_depth](img.view(1, 3, img_h, img_w))[0]
result = result.permute(1,2,0).detach().numpy().astype(float)
height, width, depth = result.shape

# Normalize tensor and obtain its PCA representation
result = normalize_data(result).reshape((-1, depth))
result = perform_PCA(result, var=99.0)[0]

# Display feature's projections along first principal component as a grayscale image
first_pc = upsample(result[...,0].reshape((height, width)), img_h, img_w)
plt.axis('off')
plt.imsave('first_principal_component.jpg', first_pc, cmap='gray')

# Plot a histogram to show the distribution of values along the first principal component
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
s = pd.Series(result[..., 0])
s.plot.hist(ax=ax, bins=50)
plt.xlabel('feature projection along first PC', fontsize=22)
plt.ylabel('count', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
fig.savefig('histogram.png', format='png', bbox_inches='tight', transparent=True)

# Cluster pixel-wise entries in the feature image use k-means. The number of clusters is determined using the Silhouette method
lower_n, upper_n = 2, 11
cluster_number_range = range(lower_n, upper_n)
max_score = max_index = -1
for idx, n_clusters in tqdm(enumerate(cluster_number_range), desc='Trying out different k\'s ...'):
    clustering_result = cluster_all(result, n_clusters=n_clusters)
    score = silhouette_score(result, clustering_result)
    max_index = idx if score>max_score else max_index
    max_score = max(score, max_score)
optimal_cluster_number = max_index + lower_n
print('Optimal number of clusters: ', optimal_cluster_number)

# Cluster using optimal clustering number
indices = cluster_all(result, n_clusters=optimal_cluster_number)
mask = indices.reshape((height, width))
upsampled_mask = upsample(mask, img_h, img_w)
plt.imsave('mask.png', upsampled_mask, cmap='cividis')