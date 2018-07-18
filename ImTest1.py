

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
img = load_img('data/train/cats/cat.8.jpg')  # this is a PIL image , tensorflow API
img.show()
print(img)

x = img_to_array(img)    # this is a Numpy array with shape ( Y, X , 3)
print(x.shape)

# x = x.astype('float32') / 255.0
print(x.dtype)
x = x.astype('int32') 
print(x.dtype)

import matplotlib.pyplot as plt 
fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.imshow(x)             # RGB type 0~255 int or 0~1 float
plt.show()


img2 = load_img('data/train/cats/cat.8.jpg',target_size=(150,150))
x2 = img_to_array(img2)    # this is a Numpy array with shape ( Y, X , 3)
print(x2.shape)
print(x2.dtype)
x2 = x2.astype('float32') / 255.0
print(x2.dtype)

fig = plt.gcf()
fig.set_size_inches(10, 5)
plt.imshow(x2)             # RGB type 0~255 int or 0~1 float
plt.show()



