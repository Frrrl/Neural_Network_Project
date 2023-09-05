import numpy as np  # 多维数组库
import tensorflow as tf  # 深度学习框架库
import matplotlib.pyplot as plt  # 画图库

"数据集"
# <editor-fold desc="创建C1和C2数据集">
dot_num = 100  # 每种类型生成100个点
x_p = np.random.normal(3., 1, dot_num)  # x轴上以3为中心，方差为1，生成100个随机正态分布的点
y_p = np.random.normal(6., 1, dot_num)  # y轴上以6为中心，方差为1，生成100个随机正态分布的点
y = np.ones(dot_num)  # 生成一个含有100个1的向量
C1 = np.array([x_p, y_p, y], dtype=np.float32).T  # C1集合，将三个向量合在一起，并转置

x_n = np.random.normal(6., 1, dot_num)  # 类似以上操作
y_n = np.random.normal(3., 1, dot_num)
y = np.zeros(dot_num)  # 生成一个含有100个0的向量
C2 = np.array([x_n, y_n, y], dtype=np.float32).T

plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')  # 绘制C1的散点图
plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')

plt.show()
data_set = np.concatenate((C1, C2), axis=0)  # 将C1和C2放在一起，作为一个数据集data_set
np.random.shuffle(data_set)  # 将data_set中的（特征-标签对）打乱重排
# </editor-fold>

"模型构造"
# <editor-fold desc="构造一个前馈神经网络">
class myFNNModel:
    def __init__(self):  # 初始化model
        self.w = tf.Variable(shape=[2, 1], dtype=tf.float32,
                             initial_value=tf.random.uniform(shape=[2, 1], minval=-0.1, maxval=0.1))
        self.b = tf.Variable(shape=[1], dtype=tf.float32, initial_value=tf.zeros(shape=[1]))
        self.trainable_variables = [self.w, self.b]

    def __call__(self, inp):  # 将Model作为函数进行调用
        logits = tf.matmul(inp, self.w) + self.b
        pred = tf.nn.sigmoid(logits)
        return pred


# </editor-fold>

"loss与accuracy的计算"
# <editor-fold desc="计算loss和accuracy的函数">
def compute_loss(pred, label):
    # <editor-fold desc="pred与label的预处理">
    if not isinstance(label, tf.Tensor):  # 判断label是否是一个Tensor
        label = tf.constant(label, dtype=tf.float32)  # 如果label不是Tensor，就将它变成一个常量Tensor
    pred = tf.squeeze(pred, axis=1)  # 给pred序列降维
    # </editor-fold>

    losses = -label * tf.math.log(pred + 1e-12) - (1 - label) * tf.math.log(1 - pred + 1e-12)
    loss = tf.reduce_mean(losses)

    # <editor-fold desc="计算accuracy">
    pred = tf.where(pred > 0.5, tf.ones_like(pred), tf.zeros_like(pred))  # 将pred中>0.5的部分替换成1，<0.5的部分换成0
    accuracy = tf.reduce_mean(tf.cast(tf.equal(label, pred), dtype=tf.float32))  # 计算模型预测的精确度
    # </editor-fold>
    return loss, accuracy


# </editor-fold>

"train一次更新参数"
# <editor-fold desc="train一次更新参数">
def train_one_step(model, optimizer, data, label):
    with tf.GradientTape() as tape:
        pred = model(data)
        loss, accuracy = compute_loss(pred, label)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, accuracy, model.w, model.b


# </editor-fold>

"training process"
# <editor-fold desc="进行200步的training">
# <editor-fold desc="获取data和label">
x_pos, y_pos, label = zip(*data_set)  # 读取数据集中的坐标和标签，分成三列
data = list(zip(x_pos, y_pos))  # 抽取出数据集中有关坐标的部分
# </editor-fold>
model = myFNNModel()
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(200):
    loss, accuracy, w, b = train_one_step(model, opt, data, label)
    if i % 10 == 0:
        print("step:", i, "loss:", loss.numpy(), "accuracy:", accuracy.numpy())

# </editor-fold>

"test the model"
# <editor-fold desc="输入一个坐标，查看其输出是多少">
cor = [(3.6, 10.5)]
pred = model(cor)
print(pred.numpy())

# </editor-fold>
