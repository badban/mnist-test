import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
def weight_variable(shape):
    #weight_initial=tf.truncated_normal(shape,stddev=0.1,dtype=tf.float32)
    weight_initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weight_initial)
def bias_variable(shape):
    #bias_initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    bias_initial=tf.constant(0.1,shape=shape)
    return tf.Variable(bias_initial)
def graph():
    # x=tf.placeholder("float",[None,784])
    # W=tf.Variable(tf.zeros([784,10]))
    # b=tf.Variable(tf.zeros([10]))
    # # y=tf.nn.softmax(tf.matmul(x,W),b)
    # y = tf.nn.softmax(tf.matmul(x, W) + b)
    # y_=tf.placeholder("float",[None,10])
    # cross_entropy=-tf.reduce_sum(y_*tf.log(y))
    #
    # train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    #
    # init=tf.initialize_all_variables()
    #
    # with tf.Session() as sess:
    #     sess.run(init)
    #     for i in range(1000):
    #         batch_xs,batch_ys=mnist.train.next_batch(100)
    #         sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    #
    #     correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    #     accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    #     print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))


    #
    # W1=weight_variable([5,5,1,32])
    # b1=bias_variable([32])
    # x=tf.placeholder("float",[None,784])
    # x_image=tf.reshape(x,[-1,28,28,1])
    # y_=tf.placeholder("float",[None,10])
    # h1_conv=tf.nn.relu(conv2d(x_image,W1)+b1)
    # h1_pool=max_pool_2x2(h1_conv)
    #
    #
    # W2=weight_variable([5,5,32,64])
    # b2=bias_variable([64])
    # h2_conv=tf.nn.relu(conv2d(h1_pool,W2)+b2)
    # h2_pool=max_pool_2x2(h2_conv)
    #
    # W_fc=weight_variable([7*7*64,1024])
    # b_fc=bias_variable([1024])
    #
    # h2_pool_reshape=tf.reshape(h2_pool,[-1,7*7*64])
    # h3_pool=tf.nn.relu(tf.matmul(h2_pool_reshape,W_fc)+b_fc)
    #
    # keep_prob=tf.placeholder("float")
    # h4_pool=tf.nn.dropout(h3_pool,keep_prob)
    #
    # W_fc2=weight_variable([1024,10])
    # b_fc2=bias_variable([10])
    # y_conv=tf.nn.softmax(tf.matmul(h4_pool,W_fc2)+b_fc2)
    #
    #
    # cross_entropy=-tf.reduce_mean(y_*tf.log(y_conv))
    #


    # train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(y_conv,1))
    # accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    #
    # with tf.InteractiveSession as sess:
    #     sess.run(tf.initialize_all_variables())
    #     for i in range(20000):
    #         batch = mnist.train.next_batch(50)
    #         if i%100==0:
    #             print(accuracy.eval(feed_dict={x:batch[0],y:batch[1],keep_prob:1.0}))
    #         train_step.run(feed_dict={x:batch[0],y:batch[1],keep_prob:1.0})

    #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



    # x = tf.placeholder("float", [None, 784])
    # y_ = tf.placeholder("float", [None, 10])
    # W_conv1 = weight_variable([5, 5, 1, 32])
    # b_conv1 = bias_variable([32])
    # x_image = tf.reshape(x, [-1, 28, 28, 1])
    #
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_pool1 = max_pool_2x2(h_conv1)
    #
    # W_conv2 = weight_variable([5, 5, 32, 64])
    # b_conv2 = bias_variable([64])
    #
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    #
    # W_fc1 = weight_variable([7 * 7 * 64, 1024])
    # b_fc1 = bias_variable([1024])
    #
    # h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #
    # keep_prob = tf.placeholder("float")
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #
    # W_fc2 = weight_variable([1024, 10])
    # b_fc2 = bias_variable([10])
    #

    x = tf.placeholder("float", [None, 784])
    y = tf.placeholder("float", [None, 10])
    x_image = tf.reshape(x,[-1,28,28,1])
    with tf.name_scope("conv1"):
        W=weight_variable([5,5,1,32])
        b=bias_variable([32])
        conv1=tf.nn.relu(conv2d(x_image,W)+b)
        maxpool1=max_pool_2x2(conv1)

    with tf.name_scope("conv2"):
        W = weight_variable([5, 5, 32, 128])
        b = bias_variable([128])
        conv2 = tf.nn.relu(conv2d(maxpool1, W) + b)
        maxpool2 = max_pool_2x2(conv2)

    with tf.name_scope("fc1"):
        W = weight_variable([7*7*128,1024])
        b = bias_variable([1024])
        #???
        input_fc = tf.reshape(maxpool2,[-1,7*7*128])
        output_fc = tf.nn.relu(tf.matmul(input_fc,W)+b)
        keep_prob = tf.placeholder("float")
        output_dropout= tf.nn.dropout(output_fc,keep_prob)

    with tf.name_scope("softmax"):
        W = weight_variable([1024,10])
        tf.summary.histogram('softmax_w', W)
        b = bias_variable([10])
        y_ = tf.nn.softmax(tf.matmul(output_dropout,W)+b)

        train(y,y_,x,keep_prob)
def train(y,y_,x,keep_prob,Train=False):
    Train=True

    cross_entropy = -tf.reduce_mean(y*tf.log(y_))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.initialize_all_variables())

    # 最大训练精度
    max_acc=0
    # 训练阶段
    if Train:


        # 合并到Summary中
        merged = tf.summary.merge_all()
        # 选定可视化存储目录
        writer = tf.summary.FileWriter("./log/mnist-log", sess.graph)

        # 保留精度最大的三代
        saver = tf.train.Saver(max_to_keep=3)
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            train_accuary = accuracy.eval(feed_dict={x:batch[0],y:batch[1],keep_prob:1.0})
            file=open("./acc.txt","a")
            if i %100 ==0:
                print("step %d, training accracy %g"%(i,train_accuary))
                result = sess.run(merged)  # merged也是需要run的
                writer.add_summary(result, i)
            if train_accuary > max_acc:
                max_acc=train_accuary
                print("train_accuary:%d"%train_accuary)
                print("max_acc:%d"%max_acc)

                saver.save(sess=sess,save_path="./model/save.ckpt",global_step=i)
            file.write(str(train_accuary) + str(" ") + str(max_acc))
            train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

            file.close()


        print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
    #验证阶段
    else:
        saver=tf.train.Saver()
        saver.restore(sess,"./model/save.ckpt")
        print("from trained model test accracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

    # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    #
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # sess=tf.InteractiveSession()
    # sess.run(tf.initialize_all_variables())
    # for i in range(20000):
    #     batch = mnist.train.next_batch(50)
    #     if i % 100 == 0:
    #         train_accuracy = accuracy.eval(feed_dict={
    #             x: batch[0], y_: batch[1], keep_prob: 1.0})
    #      print( "step %d, training accuracy %g" % (i, train_accuracy),train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}))
    #
    #
    # print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    #
    # sess.close()

if __name__=="__main__":
    graph()