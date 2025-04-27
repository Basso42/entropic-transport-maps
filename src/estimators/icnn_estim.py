from typing import Any, Callable, Sequence, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import jit
from functools import partial

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

class PositiveDense(nn.Module):
    """
    A linear transformation layer with a non-negative constraint on the weights using softplus.

    Attributes:
        dim_hidden: The number of output features.
        beta: Temperature parameter for softplus to control the smoothness/sharpness.
        kernel_init: Initializer function for the weight matrix.
    """
    dim_hidden: int
    beta: int = 30.0
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Applies the positive dense transformation to the input.

        Args:
            inputs: Input array of shape (..., input_dim).

        Returns:
            Transformed output array with non-negative weights.
        """
        kernel = self.param('kernel', self.kernel_init, (inputs.shape[-1], self.dim_hidden))
        kernel = 1 / self.beta * nn.softplus(self.beta * kernel)
        x = jax.lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())))
        return x
    

class ICNN(nn.Module):
    """
    A fully connected Input Convex Neural Network (ICNN) as proposed in Amos et al. (2017).

    This network ensures that the output is a convex function of its inputs.

    Attributes:
        dim_hidden: A sequence defining the number of units in each hidden layer.
        act_fn: The activation function to use in each layer (default: ELU).
        init_fn: Initializer for both convex and input weights.
    """
    dim_hidden: Sequence[int]
    act_fn: Callable = nn.elu
    init_fn: Callable = nn.initializers.variance_scaling(scale=1., distribution="normal", mode="fan_avg")

    def setup(self):
        """
        Initializes the ICNN layers: convex weights (w_zs) and input weights (w_ys).
        """
        self.dim = self.dim_hidden + (1,)
        self.w_zs = [PositiveDense(feature, kernel_init=self.init_fn) for feature in self.dim]
        self.w_ys = [nn.Dense(feature, use_bias=True, kernel_init=self.init_fn) for feature in self.dim]

    @nn.compact
    def __call__(self, input: Array) -> Array:
        """
        Applies the ICNN transformation to the input.

        Args:
            input: Input array of shape (..., input_dim).

        Returns:
            Output scalar array representing a convex function of the input.
        """
        y = input
        z = jnp.zeros(shape=y.shape)
        for w_z, w_y in zip(self.w_zs[:-1], self.w_ys[:-1]):
            z = self.act_fn(w_z(z) + w_y(y))
        z = self.w_zs[-1](z) + self.w_ys[-1](y)
        return z.squeeze()


class sampler_from_data:
    """
    A utility class to sample from a given dataset using JAX's random choice.

    Attributes:
        x: A dataset or array to sample from.
    """

    def __init__(self, x: Array):
        """
        Initializes the sampler with a dataset.

        Args:
            x: Array of data points to sample from.
        """
        self.x = x
        self.setup()

    def setup(self):
        """
        Sets up a JAX-jitted sample generator from the data.
        """

        @partial(jit, static_argnums=1)
        def generate_samples(key: PRNGKey, num_samples: int) -> Array:
            """
            Randomly samples `num_samples` points from the dataset using a given random key.

            Args:
                key: A JAX PRNG key.
                num_samples: Number of samples to generate.

            Returns:
                Array of sampled points from the dataset.
            """
            points = jax.random.choice(key, self.x, (num_samples,))
            return points

        self.generate_samples = generate_samples
