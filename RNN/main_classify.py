from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import codecs
import math
import random
import string
import time
import numpy as np
import unicodedata
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score
from io import open
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 优先用GPU进行训练，否则用CPU

# <editor-fold desc="预设置的各种参数">
languages = ["af", "cn", "de", "fi", "fr", "in", "ir", "pk", "za"]
all_letters = string.ascii_letters + " .,;'"
BaseDir = "C:/Users/RoxVictus/Desktop/Rnn_dataset/"   # 此处复制自己桌面上的文件夹路径
train_file_dir = BaseDir + 'train/*.txt'
val_file_dir = BaseDir + 'train/*.txt'
test_file_dir = BaseDir + '*.txt'
torch.set_num_threads(1)
os.environ['MKL_NUM_THREADS'] = str(1)
# </editor-fold>


# <editor-fold desc="训练中用到的各种函数的定义">
"1.将unicode形式的数字转化为普通的ASCII码的形式"
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


"2.读取txt文本中的每一行的信息，其中的lang参数就是指语言，返回一个list"
def getWords(baseDir, lang, train=True):
    folder = 'train/' if train else 'val/'
    with open(baseDir + folder + lang + '.txt', encoding='utf-8', errors='ignore') as file:
        line = file.read().strip().split('\n')
    return line


"3.为第2步的数据配上label"
def getLabels(lang, length):
    index = [i for i, country in enumerate(languages) if country == lang][0]
    labels = [index for _ in range(length)]
    return labels


"4.将2和3的功能进行合并，得到包含data和label的数据集"
def readData(baseDir, train=True):
    all_words = []
    all_labels = []
    for lang in languages:
        words = getWords(baseDir, lang, train=train)
        labels = getLabels(lang, len(words))
        all_words += words
        all_labels += labels
    return all_words, all_labels


"5.将一行list(包括文本、数字等)转换为pytorch可识别的tensor形式"
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, len(all_letters), dtype=torch.float)
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


"6.从多分类的结果中选择概率最高的那一个作为输出，返回值为元组，类似于（'cn',1）,调节topk()的参数可以在元组中加入多个值"
def category_from_output(output):
    top_n, top_i = output.topk(1)
    language_index = top_i[0].item()
    language = languages[language_index]
    outputs = (language, language_index)
    return outputs


"7.从数据集中随机抽取一对数据标签对用于训练"
"返回的数据为标签、数据、标签对应的one-hot_tensor和数据对应的tensor"
def random_training_pair(X, y):
    #用于从一个list中随机抽取一个元素，返回该元素与其索引值
    def random_choice(ls):
        idx = random.randint(0, len(ls) - 1)
        return ls[idx], idx

    line, idx = random_choice(X)
    line_tensor = line_to_tensor(line)
    category = y[idx]
    one_hot = np.zeros(len(languages))  # 创建一个one-hot的判别标签
    one_hot[category] = 1
    category_tensor = torch.tensor(one_hot, dtype=torch.float)
    return category, line, category_tensor, line_tensor


"8.model进行运算，给定一系列的数据（未tensor化），给出一系列分类预测结果（数字形式）"
def predict(model, X):
    predictions = []
    hidden = model.init_hidden()
    for ii in range(len(X)):
        line_tensor = line_to_tensor(X[ii])
        line_tensor = line_tensor.permute(1, 0, 2)  # 交换tensor的第一和第二维度
        output, _ = model(line_tensor, hidden=hidden)
        output = torch.max(output, 1)[1]
        predictions.append(output[0])
    return predictions


"9.给定一个model和测试集，返回模型的准确度"
def calculateAccuracy(model, X, y):
    y_pred = predict(model, X)
    return accuracy_score(y, y_pred)


"10.定义训练一幕所需要做的工作"
"一幕训练中，从数据集里随机抽取一个数据标签对，对网络参数进行一次更新"
def trainOneEpoch(model, criterion, optimizer, X, y):
    category, line, category_tensor, line_tensor = random_training_pair(X, y)
    hidden = model.init_hidden()
    optimizer.zero_grad()

    line_tensor = line_tensor.permute(1, 0, 2)
    output, _ = model(line_tensor, hidden=hidden)
    category_tensor = category_tensor.view(1, len(languages))

    loss = criterion(output, torch.max(category_tensor, 1)[1])
    loss.backward()

    optimizer.step()

    return output, loss.item(), category, line, model

# </editor-fold>

"RNN的模型"
class CharRNNClassify(nn.Module):
    def __init__(self, input_size=57, hidden_size=64, output_size=9):
        # 调用父类的init进行初始化
        super(CharRNNClassify, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        lstm_out, hidden = self.lstm(inputs.view(inputs.size(1), 1, -1))
        output = self.linear(lstm_out[-1, :, :])
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.randn(1, self.hidden_size)

# <editor-fold desc="模型进行训练的过程">
def run():
    X, y = readData(BaseDir)  # 获取训练数据集
    rnn = CharRNNClassify()  # 初始化RNN模型
    # rnn.load_state_dict(torch.load('./model_classify.pth'))  # 当使用到保存的模型时取消注释，即可加载之前训练过的模型
    criterion = nn.NLLLoss()  # loss的设定，One-hot编码的交叉熵损失函数

    n_iters = 50000  # 进行5w步的训练
    print_every = 5000  # 每5000步打印一次训练效果
    plot_every = 1000  # 每1000步存一下loss的平均结果
    lr = 1e-3  # 学习率

    current_loss = 0  # 为了画loss的曲线图设置的一系列容器
    current_val_loss = 0
    all_losses = []
    all_val_losses = []

    # 一个用于计时的函数，输入一个起始的时刻，返回当前时刻与起始时刻的时间差
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()
    optimizer = torch.optim.Adam(rnn.parameters(), lr)  # 优化器选择
    val_x, val_y = readData(BaseDir, train=False)  # 读取验证集数据

    for iter in range(1, n_iters + 1):
        output, loss, category, line, rnn = trainOneEpoch(rnn, criterion, optimizer, X, y)
        current_loss += loss  # 将loss记录下来

        # 每5000步打印一次需要的信息
        if iter % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess_i == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
                iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
            # 信息包括了当前迭代步数、进度、当前程序运行的时间、loss、用于训练的数据、模型输出的结果、正确性

        # 在验证集中对loss进行测试
        _, _, val_category_tensor, val_line_tensor = random_training_pair(val_x, val_y)
        val_line_tensor = val_line_tensor.permute(1, 0, 2)
        val_output, _ = rnn(val_line_tensor, hidden=rnn.init_hidden())
        val_category_tensor = val_category_tensor.view(1, len(languages))
        val_loss = criterion(val_output, torch.max(val_category_tensor, 1)[1])
        current_val_loss += val_loss

        # 每1000步把验证集的平均loss保存下来，并且将loss置零，用于存储下一个1000的loss
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            all_val_losses.append(current_val_loss / plot_every)
            current_loss = 0
            current_val_loss = 0

    acc = calculateAccuracy(rnn, val_x, val_y)  # 在验证集上测试准确率
    print('Validation Accuracy: ', acc)

    torch.save(rnn.state_dict(), './model_classify2.pth')  # 训练完成保存模型

    # 画出loss的曲线
    plt.figure()
    plt.plot(all_losses, 'r', label='Train')
    plt.plot(all_val_losses, 'b', label='Validate')
    plt.title('Training/Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    num_points = int(n_iters / plot_every)
    num_ticks = 5
    spacing = int(num_points / num_ticks)
    plt.xticks(np.arange(0, num_points + 1, spacing), [x * plot_every for x in range(0, num_points + 1, spacing)])
    plt.show()
# </editor-fold>

run()   # 运行此文件进行训练时去除#号，使用Name_Test文件时加上#号
