from main_classify import *
"""
该文件的作用是测试训练的RNN是否好用，输入一个名称，返回这个名称对应的概率最大的三个国家，并附上概率
"""
# <editor-fold desc="创建一个RNN模型并且从保存的模型中加载参数">
Rnn_Model = CharRNNClassify()
Rnn_Model.load_state_dict(torch.load('./model_classify.pth'))
print(all_letters)
# </editor-fold>

name = 'mbappe'

# <editor-fold desc="根据name计算出结果">
# 创建一个字典，用于将结果转化为国家
country_dict = ['Afghanistan', 'China', 'German', 'Finland', 'France', 'India', 'Iran', 'Pakistan', 'Zambia']
name_tensor = line_to_tensor(name).permute(1, 0, 2)  # 将name转化为模型可识别的tensor
print(name_tensor)
output, _ = Rnn_Model(name_tensor, hidden=Rnn_Model.init_hidden())  # 将tensor传入model中，得到结果的tensor
soft_func = nn.Softmax(dim=1)
prob = soft_func(output)  # 将结果再次进行softmax，得到每个国家对应的概率的tensor
prob_list = prob.detach().numpy()[0]  # Tensor转化为list
top_n, top_i = prob.topk(3)  # 选取出前三名对应的数值和索引号
top3_prob = top_n.detach().numpy()[0]  # 将概率转化为list
top3_index = top_i.detach().numpy()[0]  # 将编号转化为list
# </editor-fold>

# <editor-fold desc="打印各种东西">
np.set_printoptions(suppress=True)  # 不要用科学计数法
np.set_printoptions(linewidth=300)  # 打印的时候一行设置为300字符
print('The initial output of the RNN Model is :')
print(output.detach().numpy()[0], '\n')  # 打印出Model的原始输出
print('The probablity of classification is : ')
print(prob_list)
print('Afghanistan   China     German    Finland    France      India     Iran        Pakistan    Zambia', '\n')
print('The final result is :')
print("1.{} : {} %".format(country_dict[top3_index[0]], round(top3_prob[0]*100, 2)))
print("2.{} : {} %".format(country_dict[top3_index[1]], round(top3_prob[1]*100, 2)))
print("3.{} : {} %".format(country_dict[top3_index[2]], round(top3_prob[2]*100, 2)))
# </editor-fold>