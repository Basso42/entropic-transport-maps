from typing import Any, Tuple
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
from jax import jit
import optax
from optax._src import base
import flax.linen as nn
from flax.training import train_state

import ott
from ott.geometry import pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

from src.icnn import ICNN

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

class OT_Solver:
    """
    Optimal Transport Solver using an Input Convex Neural Network (ICNN) and Sinkhorn Divergence.

    This class trains a neural network to compute the optimal transport (OT) map from a source distribution
    to a target distribution using stochastic gradient descent and Sinkhorn divergence as the loss function.

    Attributes:
        num_train_iters: Number of training iterations.
        plot_function: Optional function for plotting during training.
        key: JAX PRNG key for reproducibility.
        state: Flax training state containing model parameters and optimizer.
        train_step: Jitted training step function.
    """

    def __init__(self,
                 input_dim: int,
                 neural_net: Optional[nn.Module] = None,
                 optimizer: Optional[base.GradientTransformation] = None,
                 num_train_iters: int = 20000,
                 plot_function = None,
                 seed: int = 0):
        """
        Initializes the OT_Solver.

        Args:
            input_dim: Dimensionality of the input data.
            neural_net: Optional neural network model (default is a 3-layer ICNN).
            optimizer: Optional optimizer (default is Adam with learning rate 0.001).
            num_train_iters: Number of training iterations.
            plot_function: Optional function for plotting intermediate results.
            seed: Random seed for reproducibility.
        """
        self.num_train_iters = num_train_iters
        self.plot_function = plot_function

        rng = jax.random.PRNGKey(seed)
        rng, rng_setup = jax.random.split(rng, 2)
        self.key = rng

        if optimizer is None:
            optimizer = optax.adam(learning_rate=0.001)

        if neural_net is None:
            neural_net = ICNN(dim_hidden=[64, 64, 64])

        self.setup(rng_setup, neural_net, input_dim, optimizer)

    def setup(self, rng: PRNGKey, neural_net: nn.Module, input_dim: int, optimizer: base.GradientTransformation):
        """
        Sets up the model and optimizer.

        Args:
            rng: JAX PRNG key for initialization.
            neural_net: Neural network model.
            input_dim: Input dimensionality.
            optimizer: Optax optimizer.
        """
        rng, rn_state = jax.random.split(rng, 2)
        self.state = self.create_train_state(rn_state, neural_net, optimizer, input_dim)
        self.train_step = self.get_step_fn()

    def __call__(self,
                 sampler_source: Iterator[jnp.ndarray],
                 sampler_target: Iterator[jnp.ndarray],
                 size_batch_train: int):
        """
        Runs training on the provided source and target samplers.

        Args:
            sampler_source: Iterator that generates source samples.
            sampler_target: Iterator that generates target samples.
            size_batch_train: Number of samples per training batch.

        Returns:
            The final training state.
        """
        _ = self.train_quantile_net(sampler_source, sampler_target, size_batch_train)
        return self.state

    def train_quantile_net(self,
                           sampler_source: Iterator[jnp.ndarray],
                           sampler_target: Iterator[jnp.ndarray],
                           size_batch_train: int):
        """
        Trains the neural network using source and target samplers.

        Args:
            sampler_source: Source sample generator.
            sampler_target: Target sample generator.
            size_batch_train: Batch size for training.
        """
        batch = {}
        master_key = self.key
        for step in range(self.num_train_iters):
            master_key, inner_key_source, inner_key_target = jax.random.split(master_key, num=3)
            batch['source'] = sampler_source.generate_samples(inner_key_source, size_batch_train)
            batch['target'] = sampler_target.generate_samples(inner_key_target, size_batch_train)

            self.state, loss = self.train_step(self.state, batch)

            if step % 100 == 0:
                print('Loss after {} iterations: {:.6f}'.format(step, loss))

            if step % 1000 == 0 and self.plot_function:
                master_key, inner_key_source, inner_key_target = jax.random.split(master_key, num=3)
                X_train = sampler_source.generate_samples(inner_key_source, size_batch_train)
                Y_train = sampler_target.generate_samples(inner_key_target, size_batch_train)
                self.plot_function(X_train, Y_train, self.state)

    def create_train_state(self,
                           rng: PRNGKey,
                           neural_net: nn.Module,
                           optimizer: base.GradientTransformation,
                           input_dim: int) -> train_state.TrainState:
        """
        Initializes the training state including model parameters and optimizer.

        Args:
            rng: JAX PRNG key.
            neural_net: Neural network model.
            optimizer: Optimizer for training.
            input_dim: Input dimension for initializing the model.

        Returns:
            A Flax TrainState object.
        """
        params = neural_net.init(rng, jnp.ones(input_dim))['params']
        return train_state.TrainState.create(apply_fn=neural_net.apply, params=params, tx=optimizer)

    def get_step_fn(self) -> Callable:
        """
        Creates and returns the training step function.

        Returns:
            A jitted function that performs a single training step.
        """
        def loss_fn(params, predict, batch):
            """
            Computes the Sinkhorn divergence loss between the transported source samples
            and the target samples.

            Args:
                params: Model parameters.
                predict: Model apply function.
                batch: Dictionary containing 'source' and 'target' batches.

            Returns:
                The computed loss value.
            """
            source, target = batch['source'], batch['target']
            ot_map_point = jax.grad(predict, argnums=1)
            ot_map = jax.vmap(lambda x: ot_map_point({'params': params}, x))
            predicted = ot_map(source)
            loss = sinkhorn_divergence(pointcloud.PointCloud, predicted, target,
                                       relative_epsilon=True).divergence
            return loss

        @jax.jit
        def step_fn(state: train_state.TrainState, batch: dict) -> Tuple[train_state.TrainState, float]:
            """
            Applies one optimization step using the gradient of the loss.

            Args:
                state: Current training state.
                batch: Dictionary containing 'source' and 'target'.

            Returns:
                Updated training state and computed loss.
            """
            value_and_grad_fn = jax.value_and_grad(loss_fn)
            loss, grad = value_and_grad_fn(state.params, state.apply_fn, batch)
            return state.apply_gradients(grads=grad), loss

        return step_fn
