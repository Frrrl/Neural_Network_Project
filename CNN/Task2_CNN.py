import tensorflow as tf  # 深度学习框架
from tensorflow import keras  # tensorflow的简化版API
from tensorflow.keras import optimizers, datasets  # 优化器和数据集的库
from tensorflow.keras.layers import Dense, Flatten  # 构建多层CNN需要的库
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # 构建卷积层和池化层需要的库
import matplotlib.pyplot as plt  # 画图库
import numpy as np  # 多维数组运算库

"show dataset"
# <editor-fold desc="查看图片的样式和矩阵形式">
def mnist_visualize_single(mode, idx):
    if mode == 0:
        plt.imshow(x_train[idx], cmap=plt.get_cmap('gray'))
        title = 'label=' + str(y_train[idx])
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        plt.imshow(x_test[idx], cmap=plt.get_cmap('gray'))
        title = 'label=' + str(y_test[idx])
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.show()

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
mnist_visualize_single(0, 255)
np.set_printoptions(linewidth=300)
print(x_train[255])
# </editor-fold>

"dataset"
# <editor-fold desc="从mnist数据集中导入手写体图片">
def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = x.reshape(x.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    ds = tf.data.Dataset.from_tensor_slices((x, y))  # 加载数据
    ds = ds.map(prepare_mnist_features_and_labels)  # 对数据进行预处理，主要是变为[0~1]
    ds = ds.take(20000).batch(100)  # 挑20000张图片，训练时每个step输送100张图片

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(prepare_mnist_features_and_labels)
    test_ds = test_ds.take(20000).batch(20000)
    return ds, test_ds


def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0  # 除以255，变成[0~1]的形式
    y = tf.cast(y, tf.int64)
    return x, y


# </editor-fold>

"model"
# <editor-fold desc="建立CNNmodel">
class myCNNmodel(keras.Model):
    def __init__(self):
        super(myCNNmodel, self).__init__()
        self.l1conv = Conv2D(filters=32,
                             kernel_size=(5, 5),
                             activation='relu',
                             padding='same')
        self.l2conv = Conv2D(filters=64,
                             kernel_size=(5, 5),
                             activation='relu',
                             padding='same')
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=2)
        self.flat = Flatten()
        self.dense1 = Dense(100, activation='tanh')
        self.dense2 = Dense(10)

    def call(self, inp):
        conv1 = self.l1conv(inp)
        pool1 = self.pool(conv1)
        conv2 = self.l2conv(pool1)
        pool2 = self.pool(conv2)
        flat = self.flat(pool2)
        dense1 = self.dense1(flat)
        dense2 = self.dense2(dense1)
        probs = tf.math.softmax(dense2)

        return probs


# </editor-fold>

"train and evaluate the model"
# <editor-fold desc="用数据train CNN Model">
train_ds, test_ds = mnist_dataset()  # 获取训练集数据和测试集数据
model = myCNNmodel()
opt = optimizers.Adam()
loss = 'sparse_categorical_crossentropy'
model.compile(optimizer=opt,
              loss=loss,
              metrics='accuracy')
model.fit(train_ds, epochs=2)  # 用训练集数据拟合CNN模型，进行两轮的拟合（每轮为20000/100步）
model.evaluate(test_ds)  # 用测试集数据评价模型的准确性
# </editor-fold>

"test the model"
# <editor-fold desc="测试model的准确性">
train_ds, test_ds = datasets.mnist.load_data()
imgnum = 1234  # 选取的测试图片在测试集中的编号
# <editor-fold desc="对img进行预处理">
img = np.expand_dims(test_ds[0][imgnum], axis=0)
img = np.expand_dims(img, axis=3).astype(float)
# </editor-fold>
out_matrix = model.call(img)
print(out_matrix.numpy()) # 模型给出的十分类结果
# <editor-fold desc="pause一下">
x_pause = [1, 2]
y_pause = [0, 0]
plt.plot(x_pause, y_pause)
plt.show()
# </editor-fold>
plt.imshow(test_ds[0][imgnum])  # 展示所选的图片
plt.show()
# </editor-fold>
