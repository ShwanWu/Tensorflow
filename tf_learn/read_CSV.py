# coding=utf-8

import tensorflow as tf

filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])

reader = tf.TextLineReader()
# 阅读器的read方法会输出一个key来表征输入的文件和其中的纪录(对于调试非常有用)，
# 同时得到一个字符串标量， 这个字符串标量可以被一个或多个解析器，
# 或者 转换操作 将其解码为张量并且构造成为样本。
key, value = reader.read(filename_queue)

# Default values, in case of empty columns.
# Also specifies the type of the decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
# decode_csv 操作会解析这一行内容并将其转为张量列表。
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
# col5 is the label not in feature
features = tf.concat(0, [col1, col2, col3, col4])


with tf.Session as sess:
    # start populating the filename queue
    coord = tf.train.Coordinator()
    # 来将文件名填充到队列，否则read操作会被阻塞到文件名队列中有值为止
    # 这个函数将会启动输入管道的线程，填充样本到队列中，
    # 以便出队操作可以从队列中拿到样本
    threads = tf.train.start_quene_runners(coord=coord)

    for i in range(1200):
        # retrieve a single instance
        example, label = sess.run([features, col5])

    # When done, ask the threads to stop.
    coord.request_stop()
    # And wait for them to actually do it.
    coord.join(threads)
