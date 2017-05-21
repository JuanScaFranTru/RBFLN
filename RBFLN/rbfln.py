import tensorflow as tf


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
        pass

    def _add_loss(self):
        """Add loss function to model"""
        pass

    def _add_input_and_output(self):
        """Add input and output placeholders to the model."""
        pass

    def _add_weights(self):
        """Add linear and non-linear weights to the model."""
        pass

    def _add_variance(self):
        """Compute initial values for variances to the model."""
        pass

    def _add_center_vectors(self):
        """Add center vectors nodes to the model.

        Let Q be the number of input feature vectors.
        Initialize RBF center vectors by putting v(m) = x(m) if M <= Q, else
        put v(q) = x(q) , q = 1,...,Q, and draw the remaining M - Q centers at
        random in the feature space.
        """
        pass

    def _add_ys(self):
        """Add y nodes to the model."""
        pass

    def _add_zs(self):
        """Add zs nodes to the model."""
        pass
