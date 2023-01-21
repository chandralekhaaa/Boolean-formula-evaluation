!unzip "/content/hw3_part2_data.zip"

!pip install imageio

import matplotlib.pyplot as plt
import matplotlib.image as img
import imageio as iio

def load_image(file_path):
  image = iio.imread(file_path)
  image = image /255
  return image

def initialize(image, clusters):
  image_array = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
  nrows, ncolumns = image_array.shape
  centroids = np.zeros((clusters, ncolumns))

  for i in range(clusters):
    rand1 = int(np.random.random(1)*10)
    rand2 = int(np.random.random(1)*7)
    centroids[i,0] = image_array[rand1, 0]
    centroids[i,2] = image_array[rand2, 0]
  
  return image_array, centroids

def euclidean_distance(x1, y1, x2, y2):      
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

def kmeans(image_array, centroids, clusters):
  iterations = 10
  nrows, ncolumns = image_array.shape
  index = np.zeros(nrows)
  for i in range(iterations):
    for j in range(len(image_array)):
      min_distance = 1000
      for k in range(clusters):
        x1 = image_array[j,0]
        y1 = image_array[j,1]
        x2 = centroids[k,0]
        y2 = centroids[k,1]
        if(euclidean_distance(x1,y1,x2,y2)<min_distance):
          min_distance = euclidean_distance(x1,y1,x2,y2)
          index[j] = k
  for k in range(clusters):
    sum_x = 0
    sum_y = 0
    count = 0
    for j in range(len(image_array)):
      if(index[j] == k):
        sum_x += image_array[j,0]
        sum_y += image_array[j,1]
        count += 1
  if(count == 0):
    count = 1
  centroids[k,0] = float(sum_x/count)
  centroids[k,1] = float(sum_y/count)
  return centroids, index

def compress(centroids, index, image, file_prefix, cluster, initialization):
  centroids = np.array(centroids)
  compressed_image = centroids[index.astype(int), :]
  compressed_image = np.reshape(compressed_image, image.shape)
  plt.imshow(compressed_image)
  plt.show()
  iio.imsave(os.getcwd()+"compressed_"+ file_prefix + str(cluster) + "_" + str(initialization), compressed_image, format="jpg")

def stats(original_image_path, compressed_image_path):
  org_image = os.stat(original_image_path)
  compressed_image = os.stat(compressed_image_path)
  return (org_image.st_size / float(compressed_image.st_size))

original_image_path_Koala = "/content/Koala.jpg"
original_image_path_Penguins = "/content/Penguins.jpg"

for cluster in [15, 20]:
  compression_ratios = []
  for initialization in range(1,6):
    print("Number of clusters:", cluster, "Initialization:", initialization)
    image = load_image("original_image_path_Koala")
    image_array, centroids = initialize(image, cluster)
    centroids, index = kmeans(image_array, centroids, cluster)
    compressed_image = compress(centroids, index, image, "Koala",cluster, initialization)
    ratio = stats(original_image_path_Koala, os.getcwd()+"compressed_"+ "Koala" + str(cluster) + "_" + str(initialization))
    print("Compression ratio:", ratio)
    compression_ratios.append(ratio)
  print("Average:", np.mean(compression_ratios))
  print("Variance:", np.var(compression_ratios))

for cluster in [2, 5, 10, 15, 20]:
  compression_ratios = []
  for initialization in range(1,6):
    print("Number of clusters:", cluster, "Initialization:", initialization)
    image = load_image(original_image_path_Penguins)
    image_array, centroids = initialize(image, cluster)
    centroids, index = kmeans(image_array, centroids, cluster)
    compressed_image = compress(centroids, index, image, "Penguins", cluster, initialization)
    ratio = stats(original_image_path_Penguins, os.getcwd()+"compressed_"+ "Penguins" + str(cluster) + "_" + str(initialization))
    print("Compression ratio:", ratio)
    compression_ratios.append(ratio)
  print("Average:", np.mean(compression_ratios))
  print("Variance:", np.var(compression_ratios))