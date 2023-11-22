import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("/home/vic/cs/projects/acne/Deep3DFaceRecon_pytorch/datasets/examples/000031.jpg")
mat = np.loadtxt("/home/vic/cs/projects/acne/Deep3DFaceRecon_pytorch/datasets/examples/detections/000031.txt")
print(mat)



plt.imshow(img)

#print points in image
for p in mat:
    x, y = p[0], p[1]
    plt.plot(x, y, "or", markersize=5)

plt.show()



