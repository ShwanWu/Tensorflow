# coding=utf-8
import numpy as np

# the padding method take in an input_array and add margin(zp) to it.
# applied in bp_sensitivity_map
# 边缘补灰，zp为补灰的宽度
def padding(input_array, zp):
    if zp == 0:
        return input_array
    else:
        # if the input_array is 3-D
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((
                input_depth,
                input_height + 2*zp,
                input_width + 2*zp ))
            padded_array[:,
                zp : zp + input_height,
                zp : zp + input_weight] = input_array
            return padded_array
        # if the input_array is 2-D
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[
                zp : zp + input_height,
                zp : zp + input_weight] = input_array
            return padded_array

# applied in conv()
# 根据索引i,j值选择输入数组内的被卷区域。
def get_patch(input_array, i, j, filter_width, filter_height, stride):
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[
            start_i : start_i + filter_height,
            srart_j : start_j + filter_width
        ]
    if input_array.ndim == 3:
        return input_array[:,
            start_i : start_i + filter_height,
            start_j : start_j + filter_width]

# applied in ConLayer.forward
def conv(input_array, kernel_array, output_array, stride, bias):
    # kernel_array 是一个权重矩阵。
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output.array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    # 和输入矩阵进行加权和。
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                # 在输入矩阵中顺序取区域，计算后得到加权和+偏置项的和
                get_patch(input_array, i, j, kernel_width, kernel_height,stride) * kernel_array
            ).sum() + bias

def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)

def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0,0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > max_value:
                max_value = array[i,j]
                max_i, max_j = i, j
    return max_i, max_j

# applied in ConLayer
class Filter(object):
    def __init__(self,width,height,depth):
        # randomly initiate the weights in the any filter based on its size
        # you can assume the filter is like a cube and each position has small weight that is randomly generated
        self.weights = np.random.uniform(-1e-4,1e-1,(depth,height,width))
        # with bias is the wb=0
        self.bias=0
        #return gradient [h,w,d] with every element is 1 with bias 0
        self.weights_grad =np.zeros(
            self.weights.shape
        )
        self.bias_grad=0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s'\
            % (repr(self.weights),repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def updata(self):
        # use gradient descent to update weights parameter
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias

class ConLayer(object):
    # zero_padding is what the user add to margin
    # channel_number = input_array.ndim
    # learning rate is step in adjustment of weight
    def __init__(self, input_width, input_height,
                 channel_number, filter_width, filter_height,
                 zero_padding, stride, activator, learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        # output size depend on both the input size and filter size
        self.output_width = \
            ConLayer.calculate_output_size(
                self.input_width, filter_width, zero_padding,
            stride)
        self.output_height = \
            ConvLayer.calculate_output_size(
            self.input_height, filter_height, zero_padding,
            stride)
        # initiate the output array with d*h*w
        self.output_array = np.zeros((self.filter_number,
            self.output_height,self.output_width))
        # construct filters in a list
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,
                                       filter.height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    @staticmethod
    def calculate_output_size(input_size, filter_size,
                          zero_padding, stride):
        return (input_size - filter_size +2 * zero_padding)/stride + 1

    # forward calculate of ConvLayer
    def forward(self, input_array):
        '''
        计算卷积层的输出
        输出结果保存在self.output_array
        '''
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        # 对每一个卷积核都进行conv操作。
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array,
                 filter.get_weights,
                 self.output_array[f],
                 self.stride,
                 filter.get_bias())
        # activate every element of output use 'nditer'.
        # 对矩阵进行激活。
        element_wise_op(self.output_array,self.activator.forward)

    def backward(self,input_array,sensitivity_array, activitor):
        '''
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        '''
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activitor)
        # 卷积核权值的梯度gradient求法：先求出被卷积输入矩阵的bp_sensitivity_map
        self.bp.gradient(sensitivity_array)

    def update(self):
        for filter in self.filters:
            filter.update(self, learning_rate)

    # 将误差项传递到上一层的代码实现
    def bp_sensitivity_map(self,sensitivity_array, activator):
        '''
        计算传递到上一层的sensitivity map
        sensitivity_array: 本层的sensitivity map
        activator: 上一层的激活函数
        '''
        expand_array = self.expand_sensitivity_map(sensitivity_array)
        # full卷积，对sensitivitiy map进行zero padding
        # 虽然原始输入的zero padding单元也会获得残差
        # 但这个残差不需要继续向上传递，因此就不计算了
        expand_width = expand_array.shape[2]
        zp = (self.input_width + self.filter_width - 1 - expand_width) / 2
        padded_array = padding(expand_array,zp)
        # 初始化delta_array，用于保存传递到上一层的
        # sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说，最终传递到上一层的
        # sensitivity map相当于所有的filter的
        # sensitivity map之和
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter权重翻转180度
            flipped_weights = np.array(
                map(lambda i: np.rot90(i,2), filter.get_weights)
            )
         # 计算与一个filter对应的前一个delta_array
        delta_array = self.create_delta_array()
        for d in range(delta_array.shape[0]):
            conv(padded_array[f],flipped_weights[d],delta_array[d],1,0)
        self.delta_array =+ delta_array
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array,activator.backward)
        self.delta_array *= delta_array

    # 对权重梯度对计算，即sensitivity map作为卷积核
    # 在input上进行cross-correlation
    def bp_gradient(self, sensitivity_array):
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
        for f in range(self.filter_number):
            # 计算每个权重对梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d],
                     expanded_array[f],
                     filter.weights_grad[d],1,0)
            # 计算偏置项对梯度
            filter.bias_grad = expanded_array[f].sum()

    def expand_sensitivity_map(self,sensitive_array):
        depth = sensitivity_array.shape[0]
        # when stride = 1
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding +1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding + 1)
        # construct a new sensitivity_map
        expand_array = zeros((depth,expanded_height,expanded_width))
        # copy from original sensitive_array
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:,i_pos,j_pos] = \
                    sensitive_array[:i,j]
        return expand_array

    def create_delta_array(self):
        return np.zeros(
            (self.channel_number, self.input_height, self.input_width)
        )

    @staticmethod
    def calculate_output_size(input_size,
           filter_size,zero_padding,stride):
        return (input_size - filter +
                2 * zero_padding) / stride +1

class Reluactivator(object):
    def forward(self,weight_input):
        return max(0,weight_input)

    def backward(self,output):
        return 1 if output>0 else 0

class MaxPoolingPlayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = (input_width - filter_width)/self.stride +1
        self.output_height = (input_height - filter_hright)/self.stride +1
        self.output_array = np.zeros((self.channel_number,
             self.ouytput_heioght, self.output_width))

    def forward(self, input_array):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d,i,j] = (
                        # 分层计算
                        # 对每个filter区域，
                        get_patch(input_array[d],i,j,
                        self.filter_width,
                        self.filter_height,
                        self.stride).max())

    # 对于max pooling，下一层的误差项的值会原封不动的
    # 传递到上一层对应区块中的最大值所对应的神经元，
    # 而其他神经元的误差项的值都是0。
    # 对于mean pooling，下一层的误差项的值会平均分配到
    # 上一层对应区块中的所有神经元
    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        input_array[d],i,j,
                        self.filter_width,
                        self.filter_height,
                        self.stride)
                    k, l =get_max_index(patch_array)
                    self.delta_array[d, i * self.stride + k,
                        j * self.stride + l] = \
                        sensitivity_array[d,i,j]
