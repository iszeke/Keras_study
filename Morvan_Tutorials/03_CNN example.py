import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D,MaxPool2D,Flatten
from keras.optimizers import Adam

# 下载数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据预处理
X_train = X_train.reshape(-1, 1, 28, 28) / 255
X_test = X_test.reshape(-1, 1, 28, 28) / 255

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 构建模型
model = Sequential()
# 第一层卷积
model.add(Convolution2D(
    batch_input_shape=(None,1,28,28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first'
))
model.add(Activation('relu'))
# 第一层池化
model.add(MaxPool2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first'
))

# 第二层卷积
model.add(Convolution2D(
    filters=64,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first'
))
# 第二层池化
model.add(MaxPool2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first'
))

# 全连接层1
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# 全连接层2
model.add(Dense(10))
model.add(Activation('softmax'))

# compile
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#训练
print('Training-----------------')
model.fit(X_train,y_train,batch_size=64,epochs=3)

# 测试
print('\nTesting----------------')
loss, accuracy = model.evaluate(X_test, y_test)
print('\ntest loss',loss)
print('\ntest accuracy',accuracy)

