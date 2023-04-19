# # open .npz file and print its content

# import numpy as np
# import tensorflow as tf

# with tf.io.gfile.GFile("./samples/cifar_cond_vanilla/samples_980540.npz",'rb') as f:
#     text = f.read()

# print(text)


import numpy as np

# 用你自己的文件名替换 "samples_XXXXXX.npz"
filename = "./samples/cifar_cond_vanilla/samples_980540.npz"

# 加载数据
data = np.load(filename)

# 从数据中提取 NumPy 数组
samples = data["samples"]

# 如果存在类别标签，也可以将其加载
if "label" in data:
    labels = data["label"]

print(labels)
print(samples.shape)
# 用于显示和处理图像的库
import matplotlib.pyplot as plt

# 显示第一张图像
plt.imshow(samples[0])
plt.savefig("test.png")


plt.show()