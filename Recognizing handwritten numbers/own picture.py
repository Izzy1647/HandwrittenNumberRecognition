import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window
import imageio
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


input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
training_data_file = open('/Users/zzy/Desktop/Recognizing handwritten numbers/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 10

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

print("loading ... my_own_images/2828_my_own_image.png")
img_array = imageio.imread('/Users/zzy/Desktop/test4.png', as_gray=True)

# reshape from 28x28 to list of 784 values, invert values
img_data = 255.0 - img_array.reshape(784)

# then scale data to range from 0.01 to 1.0
img_data = (img_data / 255.0 * 0.99) + 0.01
#print("min = ", numpy.min(img_data))
#print("max = ", numpy.max(img_data))

# plot image
matplotlib.pyplot.imshow(img_data.reshape(28, 28), cmap='Greys', interpolation='None')

# query the network
outputs = n.query(img_data)
print(outputs)

# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)
print("network says ", label)
