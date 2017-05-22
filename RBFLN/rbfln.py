from tensorflow.python import debug as tf_debug
import tensorflow as tf
import numpy as np


class RBFLN(object):

    def __init__(self, xs, ts, M, N, niter=100, variance=None):
        """Create a Radial Basis Functional Link Network.

        Create a RBFLN with N neurons in the input layer, M neurons in the
        hidden layer and 1 neuron in the output layer.
        The xs and ts parameters should have the same length.
        The lengths of all the elements of xs should be equal to N.
        The lengths of all the elements of ts should be equal to 1.

        :param xs: input feature vectors used for training.
        :param ts: associated output target vectors used for training.
        :param M: Number of neurons in the hidden layer.
        :param N: Number of neurons in the input layer.
        :param niter: Number of iterations.
        :param variance: The initial variance of the RBF.

        :type M: int
        :type N: int
        :type niter: int
        :type variance: float
        """
        self.xs = xs
        self.ts = ts
        self.Q = len(xs)
        self.M = M
        self.N = N
        self.niter = niter
        self.variance = variance

        # Build the neural network
        self._add_input_and_output()
        self._add_weights()
        self._add_variance()
        self._add_center_vectors()
        self._add_ys()
        self._add_zs()

        # Add the loss function as part of the model
        self._add_loss()

        # Add an optimizer: gradient descent with a step of 0.01
        self.optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train = self.optimizer.minimize(self.loss)

        # training loop
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        for i in range(niter):
            eval_loss = 0
            # TODO: Medir eficiencia de ZIP vs numpy transpose
            for x, t in zip(xs, ts):
                sess.run(self.train, {self.x: x, self.t: t})
                # evaluate training accuracy
            for x, t in zip(xs, ts):
                eval_loss += sess.run([self.loss], {self.x: x, self.t: t})
            print("Loss: ", eval_loss)

    def _add_loss(self):
        """Add loss function to model"""
        z = self.z
        t = self.t
        square = tf.square(z - t)
        self.loss = tf.reduce_sum(square)

    def _add_input_and_output(self):
        """Add input and output placeholders to the model."""
        N = self.N
        self.x = tf.placeholder(tf.float32, [N], name="Inputs")
        self.t = tf.placeholder(tf.float32, [1], name="TrueValues")

    def _add_weights(self):
        """Add linear and non-linear weights to the model."""
        # Select all weights randomly between -0.5 and 0.5
        N = self.N
        M = self.M

        self.w = tf.Variable(tf.random_uniform([N], -0.5, 0.5),
                             dtype=tf.float32, name="Linear")
        self.u = tf.Variable(tf.random_uniform([M], -0.5, 0.5),
                             dtype=tf.float32, name="Non-linear")

    def _add_variance(self):
        """Compute initial values for variances to the model."""
        N = self.N
        M = self.M
        variance = self.variance
        if variance is None:
            variance = 0.5 * (1/M) ** (1/N)

        self.variance = tf.Variable([0.5] * M,
                                    dtype=tf.float32, name="Variance")

    def _add_center_vectors(self):
        """Add center vectors nodes to the model.

        Let Q be the number of input feature vectors.
        Initialize RBF center vectors by putting v(m) = x(m) if M <= Q, else
        put v(q) = x(q) , q = 1,...,Q, and draw the remaining M - Q centers at
        random in the feature space.
        """
        M = self.M
        N = self.N
        Q = self.Q
        xs = self.xs

        v = xs[:M]

        if M > Q:
            v = np.concatenate((v, np.random.uniform(0, 1, (M - Q, N))))

        self.v = tf.Variable(v, dtype=tf.float32, name="CenterV")

    def _add_ys(self):
        """Add y nodes to the model."""
        M = self.M
        x = self.x
        v = self.v

        variance = self.variance
        r_x = tf.tile(x, [M])

        norm = tf.reshape(-tf.norm(r_x - v, axis=1), tf.shape(variance))

        self.y = tf.exp(norm ** 2.0) / (2.0 * variance)

    def _add_zs(self):
        """Add zs nodes to the model."""
        M = self.M
        N = self.N
        x = tf.reshape(self.x, [N, 1])
        y = tf.reshape(self.y, [M, 1])
        w = tf.reshape(self.w, [1, N])
        u = tf.reshape(self.u, [1, M])

        self.z = (1/(M + N)) * (tf.matmul(y, u) + tf.matmul(x, w))
