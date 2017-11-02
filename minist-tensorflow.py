# import numpy as np
# X=np.array([[1,2,3],[2,3,4]])
#
# percent=100
#
# # for _ in range(X.shape[0]):
# #    perm = np.random.permutation(X.shape[1])
# #    index = perm[:int(X.shape[1] * percent / 100)]
# #    # ind = np.random.choice(X.shape[1], int(X.shape[1] * percent / 100))
# #    X[_, index] = 0
# #
# # print(X)
#
# perm = np.random.permutation(X.shape[1])
# index = perm[:int(X.shape[1] * percent / 100)]

#
# def train():
#     cross_entropy = -tf.reduce_mean(y * tf.log(y_))
#     train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#     correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#     sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
#     sess.run(tf.initialize_all_variables())
#     for i in range(20000):
#         batch = mnist.train.next_batch(50)
#         train_accuary = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
#         if i % 100 == 0:
#             print("step %d, training accracy %g" % (i, train_accuary))
#         train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
#     print("test accuracy %g" % accuracy.eval(
#         feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
#     sess.close


# import tensorflow as tf
# def linear(input):
#     W=tf.get_variable("W",[30,10],initializer=tf.truncated_normal(mean=10,stddev=0.1))
#     b = tf.get_variable("b", [10], initializer=tf.constant(0))
#     output=tf.matmul(input,W)+b
# def mlp1():
#     input=tf.truncated_normal([30,10])
#     linear1=linear(input)
#     linear2=linear(input)
#
# def conv_relu(input, kernel_shape, bias_shape):
#     # Create variable named "weights".
#     weights = tf.get_variable("weights", kernel_shape,
#         initializer=tf.random_normal_initializer())
#     # Create variable named "biases".
#     biases = tf.get_variable("biases", bias_shape,
#         initializer=tf.constant_intializer(0.0))
#     conv = tf.nn.conv2d(input, weights,
#         strides=[1, 1, 1, 1], padding='SAME')
#     return tf.nn.relu(conv + biases)
# def my_image_filter(input_images):
#     with tf.variable_scope("conv1"):
#         # Variables created here will be named "conv1/weights", "conv1/biases".
#         relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
#     with tf.variable_scope("conv2"):
#         # Variables created here will be named "conv2/weights", "conv2/biases".
#         return conv_relu(relu1, [5, 5, 32, 32], [32])
# if __name__=="main":
#     image1 = tf.truncated_normal([100, 28, 28, 1],stddev=0.1)
#
#     image2 = tf.truncated_normal([100, 28, 28, 1], stddev=0.1)
#     result1 = my_image_filter(image1)
#     result2 = my_image_filter(image2)
    # Raises ValueError(... conv1/weights already exists ...)


import numpy as np
def erase_data(X, percent=100):
    """
    This is used for randomly earse some feature of X,
    we want to

    :param X: input feature matrix
    :param percent: percent of erased feature
    :return: erased X
    """
    X=np.array([[1,2,3],[4,5,6]])
    for _ in range(X.shape[0]):
        perm = np.random.permutation(X.shape[1])
        print(perm)
        index = perm[:int(X.shape[1] * percent / 100)]
        # ind = np.random.choice(X.shape[1], int(X.shape[1] * percent / 100))
        X[_, index] = 0
    print(X)
if __name__ == "__main__":
    X=[[1,2,3],[4,5,6]]

    erase_data(X)