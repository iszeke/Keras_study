
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# 创造一些数据
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

# plot data
plt.scatter(X, Y)
plt.show()

# 划分训练集与测试集
X_train, Y_train = X[:160], Y[:160]     #前160个点
X_test, Y_test = X[160:], Y[160:]       #后40个点

# 构建模型
model = Sequential()
model.add(Dense(units=1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')

# 训练
print('Training----------------------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost:', cost)

# 测试
print('\nTesting----------------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)

W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# 画图展示预测结果
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
