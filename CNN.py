import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 构建CNN模型
model = models.Sequential()
# 添加第一个卷积层，32个3x3的卷积核，激活函数为ReLU
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 添加最大池化层，池化窗口大小为2x2
model.add(layers.MaxPooling2D((2, 2)))
# 添加第二个卷积层，64个3x3的卷积核，激活函数为ReLU
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 添加最大池化层，池化窗口大小为2x2
model.add(layers.MaxPooling2D((2, 2)))
# 添加第三个卷积层，64个3x3的卷积核，激活函数为ReLU
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 将多维数据展平为一维向量
model.add(layers.Flatten())
# 添加全连接层，64个神经元，激活函数为ReLU
model.add(layers.Dense(64, activation='relu'))
# 添加输出层，10个神经元，激活函数为Softmax，用于10分类
model.add(layers.Dense(10, activation='softmax'))

# 打印模型结构
model.summary()

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
print("Training..................")
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
