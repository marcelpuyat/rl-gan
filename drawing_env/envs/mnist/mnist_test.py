import mnist
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
images = mnist.read()
for i in range(25):
    plt.subplot(5,5,i+1)
    im = images.next()[1]
#    plt.imshow(1 + np.ones(im.shape, dtype=int) * (im > 10))
    plt.imshow(1 + im/26)

plt.show()
