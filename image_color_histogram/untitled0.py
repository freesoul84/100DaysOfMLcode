# import the necessary packages
import matplotlib.pyplot as plt
import cv2

# load the image and show it
image = cv2.imread('aa.jpg')
#image_resize=cv2.resize(image)
cv2.imshow("image", image)

# convert the image to grayscale and create a histogram
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure(figsize=(4,4))
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("number of Pixels")
plt.plot(hist,color='g')
plt.xlim([0, 256])
plt.show()
# grab the image channels, initialize the tuple of colors,
channel = cv2.split(image)
colors = ("b", "g", "r")
plt.figure(figsize=(4,4))
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("number of Pixels")
# loop over the image channels
print(zip(channel,colors))
for (c, color) in zip(channel,colors):
	# create a histogram for the current channel and color
	hist = cv2.calcHist([c], [0], None, [256], [0, 256])
	plt.plot(hist, color = color)
	plt.xlim([0, 256]);plt.show()


#histogram
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()