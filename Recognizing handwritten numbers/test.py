'''import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,6,0.1)
y = np.sin(x)

plt.plot(x,y)
plt.show()


import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
ts = pd.Series(random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()


#import matplotlib.pyplot

import numpy
print(numpy.random.rand(3,3)-0.5)


'''
import numpy
import matplotlib.pyplot as plt
data_file = open('/Users/zzy/Desktop/Recognizing handwritten numbers/mnist_train.csv', 'r')
data_list = data_file.readlines()
data_file.close()
#print(data_list[4])
#print(len(data_list))
#print(matplotlib.get_backend())
all_values = data_list [2].split(',')
image_array = numpy.asfarray(all_values [1:]).reshape((28,28))  #文本字符串转换为实数 并创建数组 reshape成28*28的矩阵
#print(image_array)
plt.imshow( image_array, cmap='Greys', interpolation='none')
plt.show()

