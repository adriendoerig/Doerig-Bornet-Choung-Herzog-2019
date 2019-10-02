# alexnet + weights comes from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

import tensorflow as tf
from batch_maker import StimMaker, all_test_shapes
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from caffe_classes import class_names


def alexnet(TRAINING, n_batches, version, name, input_batch=None, input_checkpoint_path=None):

    ####################################################################################################################
    # Model name and logdir. Choose to train or not. Checkpoint for model saving
    ####################################################################################################################

    VERSION = str(version)
    STIM = 'shared_code'
    N_HIDDEN = 512
    save_steps = 1000  # save the model after each save_steps
    batch_size = 64
    total_n_samples = n_batches*batch_size
    noise_level = .1
    lr = 1e-6  # learning rate

    OVERFIT_UNCROWDING = False
    if OVERFIT_UNCROWDING:
        n_epochs = 100
        batches_per_epoch = 100
        n_batches = n_epochs * batches_per_epoch

    if N_HIDDEN is None:
        MODEL_NAME = name
        LOGDIR = MODEL_NAME + '_logdir/version_' + str(VERSION)
    else:
        MODEL_NAME = name
        LOGDIR = MODEL_NAME + '_logdir/mod_' + str(VERSION) + '_hidden_' + str(N_HIDDEN)
    checkpoint_path = LOGDIR + '/' + MODEL_NAME + '_hidden_' + str(N_HIDDEN) + "_model.ckpt"

    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    restore_checkpoint = not TRAINING
    continue_training_from_checkpoint = False


    ####################################################################################################################
    # Data handling (we will create data in batches later, during the training/testing)
    ####################################################################################################################

    # save parameters
    if TRAINING is True:
        filename = LOGDIR + '/' + STIM + '_training_parameters.txt'
    else:
        filename = LOGDIR + '/' + STIM + '_testing_parameters.txt'
    with open(filename, 'w') as f:
        f.write("Parameter\tvalue\n")
        variables = locals()
        variables = {key: value for key, value in variables.items()
                     if 'set' not in key}
        f.write(repr(variables))
    print('Parameter values saved.')


    ####################################################################################################################
    # Network weights and structure
    ####################################################################################################################


    x = tf.placeholder(tf.float32, [None, 227, 227, 3], name='input_image')
    tf.summary.image('input', x, 6)
    y = tf.placeholder(tf.int64, [None], name='input_label')
    is_training = tf.placeholder(tf.bool, (), name='is_training')

    net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1", allow_pickle=True).item()

    # conv1
    # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    with tf.name_scope('conv1'):
        k_h = 11
        k_w = 11
        c_o = 96
        s_h = 4
        s_w = 4
        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
        conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)
        tf.summary.histogram('conv1',conv1)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier1'+ str(VERSION),reuse=tf.AUTO_REUSE):
        classifier1 = vernier_classifier(conv1, is_training, N_HIDDEN, name='classifier1'+ VERSION)
        x_entropy1 = vernier_x_entropy(classifier1,y)
        correct_mean1 = vernier_correct_mean(tf.argmax(classifier1, axis=1), y) # match correct prediction to each entry in y
        train_op1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(x_entropy1,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier1'),
                                                       name="training_op")

    # lrn1
    # lrn(2, 2e-05, 0.75, name='norm1')
    with tf.name_scope('lrn1'):
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

    # maxpool1
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    with tf.name_scope('maxpool1'):
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv2
    # conv(5, 5, 256, 1, 1, group=2, name='conv2')
    with tf.name_scope('conv2'):
        k_h = 5
        k_w = 5
        c_o = 256
        s_h = 1
        s_w = 1
        group = 2
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)
        tf.summary.histogram('conv2',conv2)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier2'+ str(VERSION),reuse=tf.AUTO_REUSE):
        classifier2 = vernier_classifier(conv2, is_training, N_HIDDEN, name='classifier2' + VERSION)
        x_entropy2 = vernier_x_entropy(classifier2,y)
        correct_mean2 = vernier_correct_mean(tf.argmax(classifier2, axis=1), y) # match correct prediction to each entry in y
        train_op2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(x_entropy2,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier2'),
                                                       name="training_op")

    # lrn2
    # lrn(2, 2e-05, 0.75, name='norm2')
    with tf.name_scope('lrn2'):
        radius = 2
        alpha = 2e-05
        beta = 0.75
        bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

    # maxpool2
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    with tf.name_scope('maxpool2'):
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # conv3
    # conv(3, 3, 384, 1, 1, name='conv3')
    with tf.name_scope('conv3'):
        k_h = 3
        k_w = 3
        c_o = 384
        s_h = 1
        s_w = 1
        group = 1
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)
        tf.summary.histogram('conv3',conv3)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier3'+ str(VERSION),reuse=tf.AUTO_REUSE):
        classifier3 = vernier_classifier(conv3, is_training, N_HIDDEN, name='classifier3'+ VERSION)
        x_entropy3 = vernier_x_entropy(classifier3,y)
        correct_mean3 = vernier_correct_mean(tf.argmax(classifier3, axis=1), y)  # match correct prediction to each entry in y
        train_op3 = tf.train.AdamOptimizer(learning_rate=lr).minimize(x_entropy3,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier3'),
                                                       name="training_op")

    # conv4
    # conv(3, 3, 384, 1, 1, group=2, name='conv4')
    with tf.name_scope('conv4'):
        k_h = 3
        k_w = 3
        c_o = 384
        s_h = 1
        s_w = 1
        group = 2
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)
        tf.summary.histogram('conv4',conv4)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier4'+ str(VERSION),reuse=tf.AUTO_REUSE):
        classifier4 = vernier_classifier(conv4, is_training, N_HIDDEN, name='classifier4' + VERSION)
        x_entropy4 = vernier_x_entropy(classifier4,y)
        correct_mean4 = vernier_correct_mean(tf.argmax(classifier4, axis=1), y)  # match correct prediction to each entry in y
        train_op4 = tf.train.AdamOptimizer(learning_rate=lr).minimize(x_entropy4,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier4'),
                                                       name="training_op")

    # conv5
    # conv(3, 3, 256, 1, 1, group=2, name='conv5')
    with tf.name_scope('conv5'):
        k_h = 3
        k_w = 3
        c_o = 256
        s_h = 1
        s_w = 1
        group = 2
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)
        tf.summary.histogram('conv5', conv5)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier5'+ str(VERSION),reuse=tf.AUTO_REUSE):
        classifier5 = vernier_classifier(conv5, is_training, N_HIDDEN, name='classifier5'+VERSION)
        x_entropy5 = vernier_x_entropy(classifier5,y)
        correct_mean5 = vernier_correct_mean(tf.argmax(classifier5, axis=1), y)  # match correct prediction to each entry in y
        train_op5 = tf.train.AdamOptimizer(learning_rate=lr).minimize(x_entropy5,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier5'),
                                                       name="training_op")

    # maxpool5
    # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    with tf.name_scope('maxpool5'):
        k_h = 3
        k_w = 3
        s_h = 2
        s_w = 2
        padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    # fc6
    # fc(4096, name='fc6')
    with tf.name_scope('fc6'):
        fc6W = tf.Variable(net_data["fc6"][0])
        fc6b = tf.Variable(net_data["fc6"][1])
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
        tf.summary.histogram('fc6',fc6)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier6'+ str(VERSION),reuse=tf.AUTO_REUSE):
        classifier6 = vernier_classifier(fc6, is_training, N_HIDDEN, name='classifier6'+VERSION)
        x_entropy6 = vernier_x_entropy(classifier6,y)
        correct_mean6 = vernier_correct_mean(tf.argmax(classifier6, axis=1), y)  # match correct prediction to each entry in y
        train_op6 = tf.train.AdamOptimizer(learning_rate=lr).minimize(x_entropy6,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier6'),
                                                       name="training_op")

    # fc7
    # fc(4096, name='fc7')
    with tf.name_scope('fc7'):
        fc7W = tf.Variable(net_data["fc7"][0])
        fc7b = tf.Variable(net_data["fc7"][1])
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
        tf.summary.histogram('fc7', fc7)

    # vernier classifier for this layer
    with tf.variable_scope('decode_vernier7'+ str(VERSION),reuse=tf.AUTO_REUSE):
        classifier7 = vernier_classifier(fc7, is_training, N_HIDDEN, name='classifier7' +VERSION)
        x_entropy7 = vernier_x_entropy(classifier7,y)
        correct_mean7 = vernier_correct_mean(tf.argmax(classifier7, axis=1), y)  # match correct prediction to each entry in y
        train_op7 = tf.train.AdamOptimizer(learning_rate=lr).minimize(x_entropy7,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier7'),
                                                       name="training_op")

    # fc8
    # fc(1000, relu=False, name='fc8')
    with tf.name_scope('fc8'):
        fc8W = tf.Variable(net_data["fc8"][0])
        fc8b = tf.Variable(net_data["fc8"][1])
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
        tf.summary.histogram('fc8', fc8)

    with tf.variable_scope('decode_vernier8'+ str(VERSION),reuse=tf.AUTO_REUSE):
        classifier8 = vernier_classifier(fc8, is_training, N_HIDDEN, name='classifier8'+VERSION)
        x_entropy8 = vernier_x_entropy(classifier8,y)
        correct_mean8 = vernier_correct_mean(tf.argmax(classifier8, axis=1), y)  # match correct prediction to each entry in y
        train_op8 = tf.train.AdamOptimizer(learning_rate=lr).minimize(x_entropy8,
                                                       var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier8' ),
                                                       name="training_op")

    # prob
    # softmax(name='prob'))
    with tf.name_scope('prob'):
        prob = tf.nn.softmax(fc8)
        tf.summary.histogram('prob',prob)

    with tf.variable_scope('decode_vernier_prob'+ str(VERSION),reuse=tf.AUTO_REUSE):
        classifier_prob = vernier_classifier(prob, is_training, N_HIDDEN, name='classifier_prob'+VERSION)
        x_entropy_prob = vernier_x_entropy(classifier_prob,y)
        correct_mean_prob = vernier_correct_mean(tf.argmax(classifier_prob, axis=1), y)  # match correct prediction to each entry in y
        train_op_prob = tf.train.AdamOptimizer(learning_rate=lr).minimize(x_entropy_prob,
                                                           var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier_prob'),
                                                           name="training_op")


    ####################################################################################################################
    # Training
    ####################################################################################################################


    if TRAINING is True:

        # training parameters
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        summary = tf.summary.merge_all()
        update_batch_norm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        master_training_op = [train_op1, train_op2, train_op3, train_op4, train_op5, train_op6, train_op7, train_op8, train_op_prob, update_batch_norm_ops]

        input_maker = StimMaker(imSize=(227,227), shapeSize=19, barWidth=2)
        if OVERFIT_UNCROWDING:
            dataset, labels = input_maker.generate_Batch_uncrowding(batch_size*batches_per_epoch, noiseLevel=noise_level)
        else:
            training_ratios = [0, 0, 1, 0]  # 0 - vernier alone; 1- shapes alone; 2- Vernier ext; 3-vernier inside random shape; 4- vernier inside shapeMatrix

        with tf.Session() as sess:

            print('Training...')
            writer = tf.summary.FileWriter(LOGDIR+'/'+STIM+'_training', sess.graph)

            if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
                saver.restore(sess, checkpoint_path)
                print('Checkpoint found.')
            else:
                print('Training network from scratch')
                init.run()
            if (restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path) and continue_training_from_checkpoint) or not restore_checkpoint or not tf.train.checkpoint_exists(checkpoint_path):

                for iteration in range(n_batches):

                    # get data in the batches
                    if OVERFIT_UNCROWDING:
                        this_batch = (iteration%batches_per_epoch)
                        batch_data, batch_labels = dataset[this_batch*batch_size:(this_batch+1)*batch_size, :, :, :], labels[this_batch*batch_size:(this_batch+1)*batch_size]
                    else:
                        batch_data, batch_labels = input_maker.generate_Batch(batch_size, training_ratios, noiseLevel=noise_level, normalize=False, fixed_position=None)

                    if iteration % 100 == 0 and iteration % 1000 != 0:

                        # Run the training operation, measure the losses and write summary:
                        _, summ = sess.run(
                            [master_training_op, summary],
                            feed_dict={x: batch_data,
                                       y: batch_labels,
                                       is_training: TRAINING})
                        writer.add_summary(summ, iteration)

                    elif iteration % 1000 == 0 and iteration != 0:

                        # Run the training operation, measure the losses, write summary and save network:
                        _, summ = sess.run(
                            [master_training_op, summary],
                            feed_dict={x: batch_data,
                                       y: batch_labels,
                                       is_training: TRAINING})
                        writer.add_summary(summ, iteration)

                        save_path = saver.save(sess, checkpoint_path)

                    else:

                        # Run the training operation and measure the losses:
                        _ = sess.run(master_training_op,
                            feed_dict={x: batch_data,
                                       y: batch_labels,
                                       is_training: TRAINING})

                    print("\rIteration: {}/{} ({:.1f}%)\n".format(
                        iteration, n_batches,
                        iteration * 100 / n_batches),
                        end="")

                # save the model at the end
                save_path = saver.save(sess, checkpoint_path)


    ####################################################################################################################
    # Testing
    ####################################################################################################################
    SHAPES = all_test_shapes()
    shapeSize = 19
    N_tests = len(SHAPES)

    results = np.zeros((N_tests, 9))
    testing_ratios = [0, 0, 0, 1]

    if TRAINING is False:
        for i in range(N_tests):
            shapeMatrix = SHAPES[i]
            saver = tf.train.Saver()
            summary = tf.summary.merge_all()

            input_maker = StimMaker(imSize=(227, 227), shapeSize = shapeSize, barWidth=2)

            with tf.Session() as sess:

                print('\rTesting...')
                print('\rshapematrix ={}'.format(str(shapeMatrix)))
                # writer = tf.summary.FileWriter(LOGDIR+'/'+STIM+'_testing', sess.graph)
                saver.restore(sess, checkpoint_path)

                # we will collect correct responses here: one entry per vernier decoder
                correct_responses = np.zeros(shape=(9))
                # assemble the number of correct responses for each vernier decoder
                correct_mean_all = tf.stack([correct_mean1, correct_mean2, correct_mean3, correct_mean4, correct_mean5, correct_mean6, correct_mean7, correct_mean8, correct_mean_prob],axis=0, name='correct_mean_all')


                for iteration in range(n_batches):

                    if OVERFIT_UNCROWDING:
                        batch_data, batch_labels = dataset[iteration*batch_size:(iteration+1)*batch_size, :, :, :], labels[iteration*batch_size:(iteration+1)*batch_size]
                    else:
                        # get data in the batches
                        batch_data, batch_labels = input_maker.generate_Batch(batch_size, testing_ratios, noiseLevel=noise_level, normalize=False, fixed_position=None, shapeMatrix=shapeMatrix)


                    if iteration % 5 == 0:

                        # Run the training operation, measure the losses and write summary:
                        correct_in_this_batch_all, summ = sess.run(
                            [correct_mean_all, summary],
                            feed_dict={x: batch_data,
                                       y: batch_labels,
                                           is_training: TRAINING})

                    else:

                        # Run the training operation and measure the losses:
                        correct_in_this_batch_all = sess.run(correct_mean_all,
                                     feed_dict={x: batch_data,
                                                y: batch_labels,
                                                is_training: TRAINING})

                    correct_responses += np.array(correct_in_this_batch_all)

                    print("\rIteration: {}/{} ({:.1f}%)".format(iteration, n_batches, iteration * 100 / n_batches), end="")

            percent_correct = correct_responses*100/n_batches
            print('... testing done.')
            print('Percent correct for vernier decoders in ascending order: ')
            print(percent_correct)
            np.save(LOGDIR + '/' + STIM  + str(shapeMatrix), percent_correct)
            results[i, :] = percent_correct
        print('Finished testing for this model, numpy saving\n'.format())
        np.save(LOGDIR + '/' + STIM + '_results', results)
        np.save(LOGDIR + '/' + STIM + '_SHAPES', SHAPES)
        return results

    # give a batch, get xentropies out. used for occlusion experiment
    if TRAINING is 'get_batch':
        saver = tf.train.Saver()
        with tf.Session() as sess:

            saver.restore(sess, input_checkpoint_path)

            out1, out2, out3, out4, out5, out6, out7, out8, out_prob \
                = sess.run([classifier1, classifier2, classifier3, classifier4, classifier5, classifier6, classifier7, classifier8, classifier_prob],
                           feed_dict={x: input_batch,
                                      y: np.zeros(input_batch.shape[0]),
                                      is_training: False})

        return np.stack([out1, out2, out3, out4, out5, out6, out7, out8, out_prob], axis=1)


    # in case we wonder what the output of alexnet itself is.

    # for input_im_ind in range(output.shape[0]):
    #     inds = argsort(output)[input_im_ind,:]
    #     print("Image", input_im_ind)
    #     for i in range(5):
    #         print(class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]])


def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i % group == 0
    assert c_o % group == 0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


def batch_norm_layer(x, n_out, phase, name='', activation=None):
    with tf.variable_scope('batch_norm_layer', reuse=True):
        h1 = tf.layers.dense(x, n_out, activation=None, name=name)
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase, scope=name+'bn')
    if activation is None:
        return h2
    else:
        return activation(h2)

def vernier_classifier(input, is_training, n_hidden=1024, name=''):
    with tf.name_scope(name):
        batch_size = tf.shape(input)[0]

        # find how many units are in this layer to flatten it
        items_to_multiply = len(np.shape(input))-1
        n_units = 1
        for i in range(1, items_to_multiply+1):
            n_units = n_units*int(np.shape(input)[i])

        flat_input = tf.reshape(input, [batch_size, n_units])
        tf.summary.histogram('classifier_input_no_bn', flat_input)

        flat_input = tf.contrib.layers.batch_norm(flat_input, center=True, scale=True, is_training=is_training, 
                                                  scope=name + 'input_bn')
        tf.summary.histogram('classifier_input_bn', flat_input)

        if n_hidden is None:
            classifier_fc = tf.layers.dense(flat_input, 2, name='classifier_top_fc')
            tf.summary.histogram(name+'_fc', classifier_fc)
        else:
            with tf.device('/cpu:0'):
                classifier_hidden = tf.layers.dense(flat_input, n_hidden, activation=tf.nn.elu, name=name+'_hidden_fc')
                tf.summary.histogram(name+'_hidden', classifier_hidden)
            classifier_fc = tf.layers.dense(classifier_hidden, 2, activation=tf.nn.elu, name=name+'_top_fc')
            tf.summary.histogram(name+'_fc', classifier_fc)
            
        classifier_out = tf.nn.softmax(classifier_fc, name='softmax')
        
        return classifier_out


def vernier_x_entropy(prediction_vector, label):
    with tf.name_scope("x_entropy"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction_vector, labels=tf.one_hot(label,2)), name="xent")
        tf.summary.scalar("xent", xent)
        return xent


def vernier_correct_mean(prediction, label):
    with tf.name_scope('correct_mean'):
        correct = tf.equal(prediction, label, name="correct")
        correct_mean = tf.reduce_mean(tf.cast(correct, tf.float32), name="correct_mean")
        tf.summary.scalar('correct_mean', correct_mean)
        return correct_mean


def get_batch_outputs(batch, input_checkpoint_path, version, NAME='crowdmaster'):

    tf.reset_default_graph()
    batch_xentropy = alexnet('get_batch', 1, version, NAME, input_batch=batch, input_checkpoint_path=input_checkpoint_path)
    
    return batch_xentropy


if __name__=="__main__":

    NAME = 'SHARED_CODE'

    TRAINING = True
    train_n_batches = 100001

    TESTING = False
    test_n_batches = 100
    N_models = 5

    N_tests = len(all_test_shapes())

    results = np.zeros((N_models, N_tests, 9))

    for i in range(N_models):
        print('\nITERATION N°{}/{}'.format(i+1, N_models))
        tf.reset_default_graph()
        alexnet(TRAINING, train_n_batches, i, NAME)
        tf.reset_default_graph()
        results[i] = alexnet(TESTING, test_n_batches, i, NAME)

    np.save(NAME+'_logdir/final_results', results)
