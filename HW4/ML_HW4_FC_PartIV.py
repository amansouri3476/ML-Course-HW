import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm


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

x_train = np.reshape(x_train, [len(x_train), 784])
x_validation = np.reshape(x_validation, [len(x_validation), 784])

X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

n_hidden_layer = 100
learning_rate = 0.01
batch_size = 64
std = 0.1
training_iterations = 60

with tf.name_scope(name="Hidden_layer_1"):
    w1 = tf.Variable(tf.random_normal(shape=(784, n_hidden_layer), mean=0, stddev=std, seed=1), name="W")
    b1 = tf.Variable(tf.zeros([n_hidden_layer]), name="B")

    hidden_layer_in_1 = tf.matmul(X, w1) + b1
    hidden_layer_output_1 = tf.tanh(hidden_layer_in_1)

with tf.name_scope(name="Hidden_layer_2"):
    w2 = tf.Variable(tf.random_normal(shape=(n_hidden_layer, n_hidden_layer), mean=0, stddev=std, seed=1), name="W")
    b2 = tf.Variable(tf.zeros([n_hidden_layer]), name="B")

    hidden_layer_in_2 = tf.matmul(hidden_layer_output_1, w2) + b2
    hidden_layer_output_2 = tf.tanh(hidden_layer_in_2)

with tf.name_scope("output_layer"):
    w3 = tf.Variable(tf.random_normal(shape=(n_hidden_layer, 10), mean=0, stddev=std), name="W")
    b3 = tf.Variable(tf.zeros([10]), name="B")  # 10, is the number of classes since we are classifying the data into 10
    #  different classes as is mentioned in the pdf file.

    output_layer_in = tf.matmul(hidden_layer_output_2, w3) + b3
    output = tf.nn.softmax(output_layer_in)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="accuracy")

confusion_matrix = tf.confusion_matrix(tf.argmax(Y, 1), tf.argmax(output, 1))

confusion_matrix_summary = tf.summary.tensor_summary('Confusion Matrix', confusion_matrix)

train_loss_summary = tf.summary.scalar('loss_train', cross_entropy)
train_accuracy_summary = tf.summary.scalar('train_accuracy', accuracy)

validation_loss_summary = tf.summary.scalar('loss_validation', cross_entropy)
validation_accuracy_summary = tf.summary.scalar('validation_accuracy', accuracy)

merge = tf.summary.merge_all()

init = tf.global_variables_initializer()

accuracy_test_acc = 0

with tf.Session() as sess:

    sess.run(init)

    summary_writer = tf.summary.FileWriter('two_hidden_layer_loss_event')
    # ########################### Training ###########################
    for i in tqdm(range(training_iterations)):

        counter = 0

        while counter <= len(x_train)//batch_size:

            counter += 1

            x_batch, y_batch = next_batch(batch_size, x_train, y_train)
            sess.run(optimizer,  feed_dict={X: x_batch, Y: y_batch})

            summary_writer.add_summary((sess.run(train_loss_summary, feed_dict={X: x_batch, Y: y_batch})), counter)

            # if counter % 500 == 0:
            #     accuracy_train, loss_train = sess.run((accuracy, cross_entropy), feed_dict={X: x_batch, Y: y_batch})
            #     print("The train accuracy is:" + str(accuracy_train))

    # ########################### Testing ###########################

    counter = 0

    while counter <= len(x_validation):
        counter += 1

        x_batch, y_batch = next_batch(batch_size, x_validation, y_validation)
        # sess.run(optimizer, feed_dict={X: x_batch, Y: y_batch})

        summary_writer.add_summary((sess.run(validation_loss_summary, feed_dict={X: x_batch, Y: y_batch})))

    x_batch, y_batch = x_validation, y_validation
    accuracy_test, loss_test = sess.run((accuracy, cross_entropy), feed_dict={X: x_batch, Y: y_batch})
    summary_writer.add_summary((sess.run(confusion_matrix_summary, feed_dict={X: x_batch, Y: y_batch})))
    print("The test accuracy is:" + str(accuracy_test))
