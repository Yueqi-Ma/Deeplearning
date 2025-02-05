import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 加载数据集
data = load_iris()
X = data.data  # 特征
y = data.target.reshape(-1, 1)  # 标签
#load_iris()：加载鸢尾花数据集。该数据集包含 150 个样本，
#每个样本有 4 个特征（花萼长度、花萼宽度、花瓣长度、花瓣宽度），
#目标标签是 3 种鸢尾花的类别（0, 1, 2）。
#X：特征矩阵，形状为 (150, 4)。
#y：目标标签，形状为 (150, 1)。
#通过 reshape(-1, 1) 将其转换为二维数组，以便后续进行独热编码。



# 对标签进行独热编码
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#random_state=42：设置随机种子，确保每次运行代码时划分结果一致。


# 创建神经网络模型
model = Sequential([#Sequential：按顺序堆叠神经网络层。
    Dense(10, input_shape=(4,), activation='relu'),  # 输入层
    Dense(10, activation='relu'),  # 隐藏层
    Dense(3, activation='softmax')  # 输出层
])

#Dense：全连接层。
#第一层：Dense(10, input_shape=(4,), activation='relu')
#10：该层有 10 个神经元。
#input_shape=(4,)：输入数据的形状为 (4,)，即每个样本有 4 个特征。
#activation='relu'：使用 ReLU（Rectified Linear Unit）激活函数。
#第二层：Dense(10, activation='relu')
#另一个包含 10 个神经元的隐藏层，使用 ReLU 激活函数。
#第三层：Dense(3, activation='softmax')
#3：输出层有 3 个神经元，对应 3 个类别。
#activation='softmax'：使用 Softmax 激活函数，将输出转换为概率分布。


# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#optimizer='adam'：使用 Adam 优化器来更新模型参数。
#loss='categorical_crossentropy'：使用分类交叉熵作为损失函数，适用于多分类任务。
#metrics=['accuracy']：在训练过程中监控模型的准确率。


# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)