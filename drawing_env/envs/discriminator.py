from config import *
import tensorflow as tf
import numpy as np
import mnist.mnist as mnist
from ops import *

# The external API is to simply call train with a fake image. This will then do one forward prop,
# return the probability assigned to this fake image, and update its parameters.
# TODO: support larger batches
# TODO: Create new discriminator and new env that train only at ends of episodes?
class RLDiscriminator(object):

    def __init__(self, sess, input_height=MNIST_DIMENSION, input_width=MNIST_DIMENSION,
                 batch_size=1, class_labels=MNIST_DIGITS, train_class=None):
        self.sess = sess
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.class_labels = class_labels
        self.train_class = train_class # if the discriminator trains on a single class. if None, train on all
        self._build_discriminator_model()
        self.sess.run(tf.global_variables_initializer())

        # Store MNIST data
        mnist_data = mnist.read('training', 'mnist')

        # Add 1 to MAX_VAL so that if MAX_VAL divisible by NUM_POSSIBLE_PIXEL_VALUES-1,
        # we still only have NUM_POSSIBLE_PIXEL_VALUES-1 buckets for filled pixels
        divisor = int(np.ceil((mnist.MAX_VAL+1.)/(NUM_POSSIBLE_PIXEL_VALUES-1)))
        self.real_examples = {label: [] for label in self.class_labels}
        for (digit, im) in mnist_data:
            self.real_examples[digit].append((MIN_PX_VALUE + im/divisor).flatten())


    def get_fake_placeholder(self):
        return self.fake_input_images

    
    def get_fake_label_placeholder(self):
        return self.fake_input_labels

    
    def get_real_placeholder(self):
        return self.real_input_images

    
    def get_real_label_placeholder(self):
        return self.real_input_labels

    
    # TODO: Try "experience replay" for fake images (hold a buffer in memory and sample it
    # for fake batches
    def get_fake_batch(self, fake_image, label, num_unfilled):
        fake_batch = np.zeros((self.batch_size, self.input_height * self.input_width))  # TODO: Somehow do batches for fake images as well...
        fake_batch[0] = fake_image
        label_batch = np.zeros((self.batch_size, 1))
        label_batch[0] = label
        return fake_batch, label_batch

    
    def get_real_batch(self, num_unfilled):
        return self._get_next_real_batch(num_unfilled)

    
    def loss_tensors(self):
        return self.disc_real_loss, self.disc_fake_loss

    
    # Trains the discriminator's params by running a batch of real and fake images to compute
    # loss. Returns the probability the model assigned to the fake image. The closer this value
    # is to 1, this means the model is getting tricked by the fake_image into thinking it's a
    # real image.
    def train(self, fake_image, fake_label, num_unfilled, debug=False):
        real_batch, labels = self._get_next_real_batch(num_unfilled)
        fake_batch = np.zeros((self.batch_size, self.input_height * self.input_width))  # TODO: Somehow do batches for fake images as well...
        fake_batch[0] = fake_image
        fake_label_batch = np.zeros((self.batch_size, 1)) # TODO: Add more to support fake image batching
        fake_label_batch[0] = fake_label
        _, real_loss, real_prob, fake_loss, fake_prob = self.sess.run([self.train_disc, self.disc_real_loss,
                                                                       self.disc_real_probability,
                                                                       self.disc_fake_loss,
                                                                       self.disc_fake_probability],
                                                                      {self.real_input_images: real_batch,
                                                                       self.real_input_labels: labels,
                                                                       self.fake_input_images: fake_batch,
                                                                       self.fake_input_labels: fake_label_batch})
        if debug:
            print("Training")
            print("Real image: " + str(real_batch[0]))
            print("Real label: " + str(labels[0]))

        if debug:
            print("Disc loss")
            print("\tReal loss: " + str(real_loss))
            print("\tReal prob: " + str(real_prob))
            print("\tFake loss: " + str(fake_loss))
            print("\tFake prob: " + str(fake_prob))
            print("")
        return fake_prob, real_prob

    
    def get_disc_loss_batch(self, fake_images, debug=False):
        return self.sess.run([self.disc_fake_loss], {self.fake_input_images: fake_images})

    
    def get_disc_loss(self, fake_image, fake_label, num_unfilled, debug=False):
        real_batch, labels = self._get_next_real_batch(num_unfilled)
        fake_batch = np.zeros((self.batch_size, self.input_height * self.input_width))  # TODO: Somehow do batches for fake images as well...
        fake_batch[0] = fake_image
        fake_label_batch = np.zeros((self.batch_size, 1))
        fake_label_batch[0] = fake_label

        real_loss, real_prob, fake_loss, fake_prob = self.sess.run([self.disc_real_loss,
                                                                    self.disc_real_probability,
                                                                    self.disc_fake_loss,
                                                                    self.disc_fake_probability],
                                                                   {self.real_input_images: real_batch,
                                                                    self.real_input_labels: labels,
                                                                    self.fake_input_images: fake_batch,
                                                                    self.fake_input_labels: fake_label_batch})
        if debug:
            print("Disc loss, not training")
            print("\tReal loss: " + str(real_loss))
            print("\tReal prob: " + str(real_prob))
            print("\tFake loss: " + str(fake_loss))
            print("\tFake prob: " + str(fake_prob))
            print("")
        return fake_prob, real_prob

    
    # Set up all the tensors for training.
    def _build_discriminator_model(self):
        self.real_input_images = tf.placeholder(tf.float32, shape=[None, self.input_height*self.input_width],
                                                name='real_input_images')
        self.real_input_labels = tf.placeholder(tf.float32, shape=[None, 1], name='real_input_labels')
        self.fake_input_images = tf.placeholder(tf.float32, shape=[None, self.input_height*self.input_width],
                                                name='fake_input_images')
        self.fake_input_labels = tf.placeholder(tf.float32, shape=[None, 1], name='fake_input_labels')

        self.disc_real_prob, disc_real_logits = self._discriminator(self.real_input_images, self.real_input_labels)
        self.disc_fake_prob, disc_fake_logits = self._discriminator(self.fake_input_images, self.fake_input_labels, reuse=True)

        # To understand these, it's best to look at the objective function of the basic Goodfellow GAN paper.
        # TODO: Use a more sophisticated loss function with gaussian noise added, etc.
        self.disc_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_logits,
                                labels=tf.ones_like(disc_real_logits)), name="disc_real_cross_entropy")
        self.disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits,
                                labels=tf.zeros_like(disc_fake_logits)), name="disc_fake_cross_entropy")
        tf.summary.scalar("Disc_real_loss", self.disc_real_loss)
        tf.summary.scalar("Disc_fake_loss", self.disc_fake_loss)
        self.train_disc = self._optimize(self.disc_real_loss + self.disc_fake_loss)

        
    def _optimize(self, loss_tensor, learning_rate=5e-4, beta1=0.5):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
        grads = optimizer.compute_gradients(loss_tensor)
        # TODO: Tensorboard summary scalar
        return optimizer.apply_gradients(grads)

    
    def _get_next_real_batch(self, num_unfilled):
        batch = np.zeros((self.batch_size, self.input_height * self.input_width))
        labels = np.zeros((self.batch_size, 1))
        for i in xrange(self.batch_size):
            image, label = self._get_next_real_image(self.train_class)
            if num_unfilled != 0:
                image[-num_unfilled:] = UNFILLED_PX_VALUE
            batch[i] = image
            labels[i] = label
        return batch, labels

    
    # return (example, label)
    def _get_next_real_image(self, clss=None):
        if clss is None: # Sample any example from any class
            clss = np.random.choice(self.class_labels)
        examples = self.real_examples[clss]
        idx = np.random.choice(len(examples))
        return (examples[idx], clss)
        
        
    # Build the discriminator model and return the output tensor and the logits tensor.
    def _discriminator(self, image, label, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            batch_size = tf.shape(image)[0]
            reshaped_input = tf.reshape(image, tf.stack([batch_size, self.input_height, self.input_width, 1]))

            h0 = lrelu(conv2d(reshaped_input, 4, 2, 2, 1, 1, name="conv1"))
            h1 = lrelu(conv2d(h0, 8, 2, 2, 1, 1, name="conv2"))
            h2 = lrelu(conv2d(h1, 16, 2, 2, 1, 1, name="conv3"))
            h2_flatted = tf.reshape(h2, [batch_size, self.input_height * self.input_width * 16])
            concated = tf.concat([label, h2_flatted], axis=1)
            h3 = dense(concated, self.input_height * self.input_width * 2, name='dense1')
            h4 = dense(h3, 1, name='dense2')

            return tf.nn.sigmoid(h4), h4
