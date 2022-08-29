import numpy as np
import skimage.color
import skimage.io
import matplotlib.pyplot as plt


# read original image, in full color
image1 = skimage.io.imread('/home/qimaqi/Downloads/amz/amz_yolov5/data/images/00000000.png')
image2 = skimage.io.imread('/home/qimaqi/Downloads/amz/amz_yolov5/data/images/rainy.jpg')
image3 = skimage.io.imread('/home/qimaqi/Downloads/amz/amz_yolov5/data/images/00000001.png')
# display the image
fig, ax = plt.subplots()
plt.imshow(image1)
plt.show()

# tuple to select colors of each channel line
colors = ("red", "green", "blue")
channel_ids = (0, 1, 2)

# create the histogram plot, with three lines, one for
# each color
plt.figure()
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        image1[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)

plt.title("Color Histogram real rainy")
plt.xlabel("Color value")
plt.ylabel("Pixel count")

plt.show()

# tuple to select colors of each channel line
colors = ("red", "green", "blue")
channel_ids = (0, 1, 2)

# create the histogram plot, with three lines, one for
# each color
plt.figure()
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        image2[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)

plt.title("Color Histogram fake rainy")
plt.xlabel("Color value")
plt.ylabel("Pixel count")

plt.show()

# tuple to select colors of each channel line
colors = ("red", "green", "blue")
channel_ids = (0, 1, 2)

# create the histogram plot, with three lines, one for
# each color
plt.figure()
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        image3[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)

plt.title("Color Histogram original")
plt.xlabel("Color value")
plt.ylabel("Pixel count")

plt.show()