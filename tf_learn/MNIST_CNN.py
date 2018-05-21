# coding=utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_set('MNIST_data', one_hot=True)

sess = tf.InteractiveSession

# 输入图片x是一个2维的浮点数张量。这里，分配给它的shape为[None, 784]，
# 其中784是一张展平的MNIST图片的维度。None表示其值大小不定，
# 在这里作为第一个维度值，用以指代batch的大小，意即x的数量不定。
# 输出类别值y_也是一个2维张量，其中每一行为一个10维的one-hot向量,用于代表对应某一MNIST图片的类别。
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# 我们在调用tf.Variable的时候传入初始值。在这个例子里，我们把W和b都初始化为零向量。
# W是一个784x10的矩阵（因为我们有784个特征和10个输出值）。
# b是一个10维的向量（因为我们有10个分类）。
W = tf.Variable(tf.zeros[784, 10])
b = tf.Variable(tf.zeros[10])

sess.run(tf.initialize_all_variables())  # 括号内为变量初始化op

# 实现我们的回归模型:
# 然后计算每个分类的softmax概率值，softmax模型可以用来给不同的对象分配概率
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 我们计算的交叉熵是指整个minibatch的，对于100个数据点的预测表现比单一数据点的表现能更好地描述模型的性能。
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 这一行代码实际上是用来往计算图上添加一个新操作，其中包括计算梯度，
# 计算每个参数的步长变化，并且计算出新的参数值。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
# 每一步迭代，我们都会加载50个训练样本，然后执行一次train_step，
# 并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。

# 评估准确性
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(accuracy=tf.reduce_mean())
print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})

# 建一个多层卷积网络
def weight_variable(shape):
    innitial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(innitial)

def bias_variable(shape):
    innitial = tf.constant(0.1, shape=shape)
    return tf.Variable(innitial)

# 卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='same')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='same')

# 我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。
# 卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]，
# 前两个维度是patch的大小5x5，接着是输入的通道数目1，最后是输出的通道数目32。
# 第一层：
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层：
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连阶层：
# 图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout层
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc_2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc_2) + b_fc2)


