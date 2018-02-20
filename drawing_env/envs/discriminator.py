import tensorflow as tf
import numpy as np
from ops import *

# The external API is to simply call train with a fake image. This will then do one forward prop,
# return the probability assigned to this fake image, and update its parameters.
class RLDiscriminator(object):

    def __init__(self, sess, input_height=3, input_width=3, batch_size=1):
        self.sess = sess
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self._build_discriminator_model()
        self.sess.run(tf.global_variables_initializer())

    # Trains the discriminator's params by running a batch of real and fake images to compute
    # loss. Returns the probability the model assigned to the fake image. The closer this value
    # is to 1, this means the model is getting tricked by the fake_image into thinking it's a
    # real image.
    def train(self, fake_image, debug=False):
        real_batch = self._get_next_real_batch()
        fake_batch = np.zeros((1, self.input_height * self.input_width))  # TODO: Somehow do batches for fake images as well...
        fake_batch[0] = fake_image
        _, real_loss, real_prob, fake_loss, fake_prob = self.sess.run([self.train_disc, self.disc_real_loss, self.discriminator_real_probability, self.disc_fake_loss, self.discriminator_fake_probability], {self.real_input_images: real_batch, self.fake_input_images: fake_batch})
        if debug:
            print("Disc loss")
            print("\tReal loss: " + str(real_loss))
            print("\tReal prob: " + str(real_prob))
            print("\tFake loss: " + str(fake_loss))
            print("\tFake prob: " + str(fake_prob))
            print("")
        return fake_prob, real_prob

    def get_disc_loss(self, fake_image, debug=False):
        real_batch = self._get_next_real_batch()
        fake_batch = np.zeros((1, self.input_height * self.input_width))  # TODO: Somehow do batches for fake images as well...
        fake_batch[0] = fake_image
        real_loss, real_prob, fake_loss, fake_prob = self.sess.run([self.disc_real_loss, self.discriminator_real_probability, self.disc_fake_loss, self.discriminator_fake_probability], {self.real_input_images: real_batch, self.fake_input_images: fake_batch})
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
        self.real_input_images = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_height*self.input_width], name='real_input_images')
        self.fake_input_images = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_height*self.input_width], name='fake_input_images')

        self.discriminator_real_probability, discriminator_real_logits = self._discriminator(self.real_input_images)
        self.discriminator_fake_probability, discriminator_fake_logits = self._discriminator(self.fake_input_images, reuse=True)

        # To understand these, it's best to look at the objective function of the basic Goodfellow GAN paper.
        # TODO: Use a more sophisticated loss function with gaussian noise added, etc.
        self.disc_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_real_logits, labels=tf.ones_like(discriminator_real_logits)), name="disc_real_cross_entropy")
        self.disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_fake_logits, labels=tf.zeros_like(discriminator_fake_logits)), name="disc_fake_cross_entropy")

        self.train_disc = self._optimize(self.disc_real_loss + self.disc_fake_loss)

    def _optimize(self, loss_tensor, learning_rate=1e-4, beta1=0.5):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1)
        grads = optimizer.compute_gradients(loss_tensor)
        # TODO: Tensorboard summary scalar
        return optimizer.apply_gradients(grads)

    def _get_next_real_batch(self):
        batch = np.zeros((self.batch_size, self.input_height * self.input_width))
        for i in xrange(self.batch_size):
            batch[i] = self._get_next_real_image()
        return batch

    def _get_next_real_image(self):
        # For now, this is just a 4x4 image with a black dot in the middle.
        return np.array([1, 1, 1, 1, \
                         1, 2, 2, 1, \
                         1, 2, 2, 1, \
                         1, 1, 1, 1])

    # Build the discriminator model and return the output tensor and the logits tensor.
    def _discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # noisy_image = gaussian_noise_layer(image)
            # TODO: Convert this into an actual ConvNet
            h0 = lrelu(dense(image, 32, name="dense1"))
            h1 = lrelu(dense(h0, 32, name="dense2"))
            h2 = dense(h1, 1, name="dense3")

            return tf.nn.sigmoid(h2), h2