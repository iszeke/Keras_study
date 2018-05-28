import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

# 下载数据
(x_train,_), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255 - 0.5
x_test = x_test.astype('float32') / 255 - 0.5
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

# 自编码器
encoding_dim = 2
input_img = Input(shape=(784,))

# 编码层
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoded_output = Dense(encoding_dim)(encoded)

# 解码层
decoded = Dense(10, activation='relu')(encoded_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

# 构建自编码器
autoencoder = Model(input=input_img, output=decoded)
# 构建一个编码器用于画图
encoder = Model(input=input_img, output=encoded_output)

# 编译
autoencoder.compile(optimizer='adam',loss='mse')

# 训练
autoencoder.fit(x_train,x_train,epochs=20,batch_size=256,shuffle=True)

# 画图
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()













