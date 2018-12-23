import numpy
import scipy.special
import matplotlib.pyplot as plt
class neuralNetwork:
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        # 初始化输入层、隐藏层、输出层、学习率
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.learn_rate = learning_rate
        # 初始化权重矩阵
        self.w_matrix_input_to_hidden = (numpy.random.rand(self.hnodes, self.inodes)-0.5)
        self.w_matrix_hideen_to_output = (numpy.random.rand(self.onodes,self.hnodes)-0.5)
        self.activation_function = lambda x: scipy.special.expit(x)  # 匿名函数 调用sigmoid函数
        pass
    def train(self,input_list,target_list):
        inputs = numpy.array(input_list, ndmin=2).T  # 把输入的列表转化为数组并转制
        targets = numpy.array(target_list,ndmin=2).T  # 把目标列表转化为数组并转制
        hidden_inputs = numpy.dot(self.w_matrix_input_to_hidden, inputs)  # 中间层输入
        hidden_outputs = self.activation_function(hidden_inputs)  # 中间层输出
        final_inputs = numpy.dot(self.w_matrix_hideen_to_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)  # 输出层输出
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.w_matrix_hideen_to_output.T,output_errors)
        # 更新中间层到输出层、输入层到中间层的权重矩阵
        self.w_matrix_hideen_to_output += self.learn_rate * numpy.dot((output_errors * final_outputs * (1-final_outputs)),numpy.transpose(hidden_outputs))
        self.w_matrix_input_to_hidden += self.learn_rate *numpy.dot((hidden_errors * hidden_outputs * (1-hidden_outputs)),numpy.transpose(inputs))
        pass
    def query(self,inputs_list):
        # 计算输出
        inputs = numpy.array(inputs_list,ndmin=2).T  # 把输入的列表转化为数组并转制
        hidden_inputs = numpy.dot(self.w_matrix_input_to_hidden,inputs)  # 中间层输入
        hidden_outputs = self.activation_function(hidden_inputs)  # 中间层输出
        final_inputs = numpy.dot(self.w_matrix_hideen_to_output,hidden_outputs)  # 输出层传入信号
        final_outputs = self.activation_function(final_inputs)  # 输出层输出
        return final_outputs

# 准备训练数据集

#print(data_list[4])
#print(len(data_list))
#single_digit_values = data_list[0].split(',')  # 逗号分割
#image_array = numpy.asfarray(single_digit_values [1:]).reshape((28,28))  #文本字符串转换为实数 并创建数组 reshape成28*28的矩阵
#print(image_array)
#plt.imshow( image_array, cmap='Greys', interpolation='none')
#plt.show()

#scale_controlled_input = (numpy.asfarray(single_digit_values[1:]) / 255.0 * 0.99) + 0.01  #
#print(scale_controlled_input)

input_nodes = 784
hidden_nodes = 130
output_nodes = 10
learning_rate = 0.3

#神经网络对象n
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

train_data_file = open('/Users/zzy/Desktop/Recognizing handwritten numbers/mnist_train_100.csv', 'r')
train_data_list = train_data_file.readlines()
train_data_file.close()



#学习世代
epochs = 5

for e in range(epochs):
    for record in train_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# 准备测试数据集
test_data_file = open('/Users/zzy/Desktop/Recognizing handwritten numbers/mnist_test_10.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()
#single_digit_values = test_data_list[0].split(',')
#print(single_digit_values[0])
#print(n.query((numpy.asfarray(single_digit_values[1:])/255.0*0.99)+0.01))
#score = 0

scoreboard = []

for record in test_data_list:
    digit_data = record.split(',')
    correct_result = digit_data[0]
    inputs = (numpy.asfarray(digit_data[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    result = numpy.argmax(outputs)
    if (result == correct_result):
        scoreboard.append(1)
    else:
        scoreboard.append(0)
scoreboard_array = numpy.asfarray(scoreboard)
print(scoreboard_array)

performance = scoreboard_array.sum() / scoreboard_array.size

print('performance:',performance)

