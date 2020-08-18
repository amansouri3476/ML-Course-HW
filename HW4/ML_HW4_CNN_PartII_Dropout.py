# Python 2.7
# By: Amin Mansouri

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split

tf.reset_default_graph()

# image data input (img shape: 28*28)
n_input = 28

n_classes = 10
std = 0.1
epsilon = 1e-4

training_iterations = 10
learning_rate = 0.001
batch_size = 64
keep_prob_input = 0.8

weights = {
    'weight_conv_layer_1': tf.get_variable('W0', shape=(5, 5, 1, 64),
                                           initializer=tf.initializers.random_normal(stddev=std)
                                           ),
    'weight_conv_layer_2': tf.get_variable('W1', shape=(5, 5, 64, 64),
                                           initializer=tf.initializers.random_normal(stddev=std)),
    'weight_conv_layer_3': tf.get_variable('W2', shape=(5, 5, 64, 128),
                                           initializer=tf.initializers.random_normal(stddev=std)),
    'weight_conv_layer_4': tf.get_variable('W3', shape=(5, 5, 128, 128),
                                           initializer=tf.initializers.random_normal(stddev=std)),

    'w_fc': tf.get_variable('W_fc', shape=(7 * 7 * 128, 256), initializer=tf.initializers.random_normal(stddev=std)),
    'out': tf.get_variable('W_out', shape=(256, n_classes), initializer=tf.initializers.random_normal(stddev=std)),
}
biases = {
    'bias_conv_layer_1': tf.get_variable('B0', shape=64, initializer=tf.initializers.random_normal(stddev=std)),
    'bias_conv_layer_2': tf.get_variable('B1', shape=64, initializer=tf.initializers.random_normal(stddev=std)),
    'bias_conv_layer_3': tf.get_variable('B2', shape=128, initializer=tf.initializers.random_normal(stddev=std)),
    'bias_conv_layer_4': tf.get_variable('B3', shape=128, initializer=tf.initializers.random_normal(stddev=std)),
    # 'bias_conv_layer_3': tf.get_variable('B2', shape=128, initializer=tf.contrib.layers.xavier_initializer()),
    'b_fc': tf.get_variable('B_fc', shape=256, initializer=tf.initializers.random_normal(stddev=std)),
    'out': tf.get_variable('B_out', shape=n_classes, initializer=tf.initializers.random_normal(stddev=std)),
}


# noinspection PyShadowingNames
def conv2d(x, w, b, strides=1):
    # Conv2D wrapper, with bias and ReLU activation
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def next_batch(num, data, labels):

    """Return a total of `num` random samples and labels."""
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


pickle_in = open('train.data', 'rb')
train_data = pickle.load(pickle_in)

pickle_in = open('test.data', 'rb')
test_data = pickle.load(pickle_in)

x_train, x_validation, y_train, y_validation = train_test_split(train_data['data'], train_data['labels'], test_size=0.2)

# both placeholders are of type float
x = tf.placeholder("float", [None, 28, 28, 1])
# x = tf.placeholder("float", [28, 28])
y = tf.placeholder("float", [None, n_classes])

keep_prob = tf.placeholder(tf.float32)

# Dropout
x = tf.nn.dropout(x, keep_prob=keep_prob_input)

# here we call the conv2d function we had defined above and pass the input image x, weights weight_conv_layer_1
# and bias bias_conv_layer_1.
conv1 = conv2d(x, weights['weight_conv_layer_1'], biases['bias_conv_layer_1'], strides=2)
# Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
conv1_maxpool = maxpool2d(conv1, k=1)
batch_mean_1, batch_var_1 = tf.nn.moments(conv1_maxpool, [0])
beta_1 = tf.Variable(tf.zeros(conv1_maxpool.get_shape().as_list()[1:]))
scale_1 = tf.Variable(tf.zeros(conv1_maxpool.get_shape().as_list()[1:]))
conv1_maxpool = tf.nn.batch_normalization(conv1_maxpool, batch_mean_1, batch_var_1, offset=beta_1, scale=scale_1, variance_epsilon=epsilon)

# Dropout
conv1_maxpool = tf.nn.dropout(conv1_maxpool, keep_prob=keep_prob)

conv2 = conv2d(conv1_maxpool, weights['weight_conv_layer_2'], biases['bias_conv_layer_2'], strides=2)
conv2_maxpool = maxpool2d(conv2, k=1)
batch_mean_2, batch_var_2 = tf.nn.moments(conv2_maxpool, [0])
beta_2 = tf.Variable(tf.zeros(conv2_maxpool.get_shape().as_list()[1:]))
scale_2 = tf.Variable(tf.zeros(conv2_maxpool.get_shape().as_list()[1:]))
conv2_maxpool = tf.nn.batch_normalization(conv2_maxpool, batch_mean_2, batch_var_2, offset=beta_2, scale=scale_2, variance_epsilon=epsilon)

# Dropout
conv2_maxpool = tf.nn.dropout(conv2_maxpool, keep_prob=keep_prob)

conv3 = conv2d(conv2_maxpool, weights['weight_conv_layer_3'], biases['bias_conv_layer_3'], strides=1)
batch_mean_3, batch_var_3 = tf.nn.moments(conv3, [0])
beta_3 = tf.Variable(tf.zeros(conv3.get_shape().as_list()[1:]))
scale_3 = tf.Variable(tf.zeros(conv3.get_shape().as_list()[1:]))
conv3 = tf.nn.batch_normalization(conv3, batch_mean_3, batch_var_3, offset=beta_3, scale=scale_3, variance_epsilon=epsilon)

# Dropout
conv3 = tf.nn.dropout(conv3, keep_prob=keep_prob)

conv4 = conv2d(conv3, weights['weight_conv_layer_4'], biases['bias_conv_layer_4'], strides=1)
batch_mean_4, batch_var_4 = tf.nn.moments(conv4, [0])
beta_4 = tf.Variable(tf.zeros(conv3.get_shape().as_list()[1:]))
scale_4 = tf.Variable(tf.zeros(conv4.get_shape().as_list()[1:]))
conv4 = tf.nn.batch_normalization(conv4, batch_mean_4, batch_var_4, offset=beta_4, scale=scale_4, variance_epsilon=epsilon)

# Dropout
conv4 = tf.nn.dropout(conv4, keep_prob=keep_prob)

# Fully connected layer
# Reshape conv4 output to fit fully connected layer input
fc1 = tf.reshape(conv4, [-1, weights['w_fc'].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, weights['w_fc']), biases['b_fc'])
fc1 = tf.nn.relu(fc1)
# Output, class prediction
# finally we multiply the fully connected layer with the weights and add a bias term.
out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

curr_img = np.reshape(x_train[1], (28, 28))
curr_lbl = np.argmax(y_train[1, :])
plt.imshow(curr_img)
plt.show()

# Reshape training and testing image
train_X = x_train.reshape(-1, 28, 28, 1)
test_X = x_validation.reshape(-1, 28, 28, 1)

print([train_X.shape, test_X.shape])

train_y = y_train
test_y = y_validation

print(train_y.shape, test_y.shape)

prediction = out

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image.
#  and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

confusion_matrix = tf.confusion_matrix(tf.argmax(prediction, 1), tf.argmax(y, 1), num_classes=10)
# calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# summary settings:

train_loss_summary = tf.summary.scalar('loss_train', cost)
train_accuracy_summary = tf.summary.scalar('train_accuracy', accuracy)
train_summaries = tf.summary.merge([train_loss_summary, train_accuracy_summary])
# file_writer = tf.summary.FileWriter('./Output', sess.graph)

validation_loss_summary = tf.summary.scalar('loss_validation', cost)
validation_accuracy_summary = tf.summary.scalar('validation_accuracy', accuracy)
validation_summaries = tf.summary.merge([validation_loss_summary, validation_accuracy_summary])
# validation_file_writer = tf.summary.FileWriter(output_folder)

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Batch Mode Learning

counter = 0
disp_counter = 20

mis_classified_idx = []

with tf.Session() as sess:
    sess.run(init)

    merge = tf.summary.merge_all()
    summary_writer_train = tf.summary.FileWriter('./Output_train', sess.graph)
    summary_writer_validation = tf.summary.FileWriter('./Output_validation', sess.graph)
    summary_writer_conv_layer_1 = tf.summary.FileWriter('./conv_1_filters')
    summary_writer_conv_layer_2 = tf.summary.FileWriter('./conv_2_filters')

    # ########################################## Training ########################################## #

    for i in range(training_iterations):

        for batch in tqdm(range(len(train_X) // batch_size // training_iterations)):

            counter += 1
            batch_x = train_X[(batch + i * len(train_X) // batch_size // training_iterations) *
                              batch_size:min((batch + 1 + i * len(train_X) // batch_size // training_iterations) *
                                             batch_size, len(train_X))]
            batch_y = train_y[(batch + i * len(train_X) // batch_size // training_iterations) *
                              batch_size:min((batch + 1 + i * len(train_X) // batch_size // training_iterations) *
                                             batch_size, len(train_y))]

            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                 y: batch_y, keep_prob: 0.5})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y, keep_prob: 0.5})

            summary_writer_train.add_summary(
                (sess.run(train_summaries, feed_dict={x: batch_x,
                                                      y: batch_y, keep_prob: 0.5})), batch)

            if counter % disp_counter == 0:

                print("Iter " + str(i) + ", Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
                      )
                print("Optimization Finished!")

    # ########################################## Testing ########################################## #

    counter = 0

    while counter <= len(x_validation):
        counter += 1

        x_batch, y_batch = next_batch(1, x_validation, y_validation)
        summary_writer_validation.add_summary((sess.run(validation_loss_summary, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})))

        prediction_result = sess.run(correct_prediction, feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
        prediction_result = np.bool(prediction_result)
        # print(type(prediction_result))
        # print(prediction_result)
        if prediction_result:
            # print('hello')
            mis_classified_idx.append(counter)
            # print('prediction for valid id: ' + str(counter) + 'is: ' + str(prediction_result) + ' which is FALSE')

    # Calculate accuracy for all test images
    x_batch, y_batch = x_validation, y_validation
    accuracy_test, loss_test = sess.run((accuracy, cost), feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})
    print("The test accuracy is:" + str(accuracy_test))

    # summary_writer_conv_layer_1.add_summary(sess.run(filter_summary_conv_1, feed_dict={x: test_X,
    #                                                                                    y: test_y}), i)
    # summary_writer_conv_layer_2.add_summary(sess.run(filter_summary_conv_2, feed_dict={x: test_X,
    #                                                                                    y: test_y}), i)

    # Plotting first layer's weights

    weights_image_conv_1 = np.reshape(sess.run(weights['weight_conv_layer_1']).T, newshape=(64, 5, 5))

    fig_1 = plt.figure()
    plt.figure(num=None, figsize=(30, 30), dpi=100)
    plt.title('First Convolution Layer Weights')

    for j in np.arange(0, 64):
        plt.subplot(8, 8, j + 1)
        plt.imshow((weights_image_conv_1[j][:][:]).T)

    plt.show()

    # Plotting first cnn layer's output for arbitrary images

    feed_x = test_X[15]
    feed_x = feed_x.reshape(-1, 28, 28, 1)
    first_layer_output_class_5 = np.reshape(sess.run(conv1, feed_dict={x: feed_x}).T, newshape=(64, 14, 14))

    fig_2 = plt.figure()
    plt.figure(num=None, figsize=(30, 30), dpi=100)
    plt.title('First Convolution Layer Outputs')

    for j in np.arange(0, 64):
        plt.subplot(8, 8, j + 1)
        plt.imshow((first_layer_output_class_5[j][:][:]).T)

    plt.show()

    feed_x = test_X[150]
    feed_x = feed_x.reshape(-1, 28, 28, 1)
    first_layer_output_class_5 = np.reshape(sess.run(conv1, feed_dict={x: feed_x}).T, newshape=(64, 14, 14))

    fig_2 = plt.figure()
    plt.figure(num=None, figsize=(30, 30), dpi=100)
    plt.title('First Convolution Layer Outputs')

    for j in np.arange(0, 64):
        plt.subplot(8, 8, j + 1)
        plt.imshow((first_layer_output_class_5[j][:][:]).T)

    plt.show()

    feed_x = test_X[1500]
    feed_x = feed_x.reshape(-1, 28, 28, 1)
    first_layer_output_class_5 = np.reshape(sess.run(conv1, feed_dict={x: feed_x}).T, newshape=(64, 14, 14))

    fig_2 = plt.figure()
    plt.figure(num=None, figsize=(30, 30), dpi=100)
    plt.title('First Convolution Layer Outputs')

    for j in np.arange(0, 64):
        plt.subplot(8, 8, j + 1)
        plt.imshow((first_layer_output_class_5[j][:][:]).T)

    plt.show()

    summary_writer_train.close()
    summary_writer_validation.close()

# For future use: tensorboard --logdir=train:./Output_train,validation:./Output_validation,conv_1:./conv_1_filters,
# conv2:./conv_2_filters
