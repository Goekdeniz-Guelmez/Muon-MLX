# Copyright © 2023-2024 Apple Inc.

import math
from typing import Callable, List, Optional, Tuple, Union

import mlx.core as mx
from mlx.nn import Module
from mlx.utils import tree_map, tree_reduce


class Optimizer:
    """The base class for all optimizers. It allows us to implement an
    optimizer on a per-parameter basis and apply it to a parameter tree.
    """

    def __init__(self, schedulers=None):
        self._initialized = False
        self._state = {"step": mx.array(0, mx.uint64)}
        self._schedulers = {k: v for k, v in (schedulers or {}).items()}

    def update(self, model: Module, gradients: dict):
        """Apply the gradients to the parameters of the model and update the
        model with the new parameters.

        Args:
            model (mlx.nn.Module): An mlx module to be updated.
            gradients (dict): A Python tree of gradients, most likely computed
                              via :func:`mlx.nn.value_and_grad`.
        """
        model.update(self.apply_gradients(gradients, model))

    def init(self, parameters: dict):
        """Initialize the optimizer's state

        This function can be used to initialize optimizers which have state
        (like momentum in :class:`SGD`). Using this method is optional as the
        optimizer will initialize itself if the state is not yet set. However,
        there are some cases where explicit initialization is useful in order
        to have access to the :attr:`Optimizer.state` before the first call to
        :meth:`Optimizer.update`.

        Args:
            model (dict): A Python tree of parameters.

        Example:
            >>> optimizer = optim.SGD(learning_rate=1e-1, momentum=0.9)
            >>> model = nn.Linear(2, 2)
            >>> optimizer.init(model.trainable_parameters())
            >>> optimizer.state.keys()
            dict_keys(['step', 'learning_rate', 'weight', 'bias'])
        """

        # Iniatilize the optimizer state to match the parameter state
        def update_state(params, state):
            if isinstance(params, (list, tuple)):
                state = list(state)
                for i in range(len(state)):
                    state[i] = update_state(params[i], state[i])
                if len(state) != len(params):
                    state.extend(tree_map(lambda x: {}, params[len(state) :]))
                return type(params)(state)
            elif isinstance(params, dict):
                for k, v in params.items():
                    if k not in state:
                        state[k] = tree_map(lambda x: {}, v)
                    else:
                        state[k] = update_state(v, state[k])
                return state
            else:
                return state

        update_state(parameters, self._state)
        tree_map(lambda p, s: s or self.init_single(p, s), parameters, self._state)
        self._initialized = True

    def init_single(self, parameter: mx.array, state: dict):
        """To be extended by the children classes to implement each optimizer's
        state initialization.

        Args:
            parameter (mx.array): A single parameter that will be optimized.
        """
        raise NotImplementedError()

    def apply_gradients(self, gradients: dict, parameters: dict):
        """Apply the gradients to the parameters and return the updated parameters.

        Can be used to update a model via
        ``model.update(opt.apply_gradients(grads, model))`` which is precisely
        how :meth:`Optimizer.update` is implemented.

        Args:
            gradients (dict): A Python tree of gradients.
            parameters (dict): A Python tree of parameters. It can be a
              superset of the gradients. In that case the returned python
              tree will be of the same structure as the gradients.
        """
        if not self._initialized:
            self.init(gradients)

        # Update any scheduled variables
        for param, scheduler in self._schedulers.items():
            self.state[param] = scheduler(self.step)

        # Increment the step
        self.state["step"] = self.step + 1

        # Apply the update
        return tree_map(self.apply_single, gradients, parameters, self.state)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """To be extended by derived classes to implement the optimizer's update.

        Args:
            gradient (mx.array): The ``parameter`` gradient.
            parameter (mx.array): The ``parameter`` to update.
            state (dict): The optimizer's state.
        """
        raise NotImplementedError()

    @property
    def state(self):
        """The optimizer's state dictionary."""
        return self._state

    @state.setter
    def state(self, state: dict):
        self._initialized = False
        self._state = state

    @property
    def step(self):
        return self.state["step"]

    @property
    def learning_rate(self):
        return self.state["learning_rate"]

    @learning_rate.setter
    def learning_rate(self, learning_rate: Union[float, mx.array]):
        self.state["learning_rate"] = mx.array(learning_rate)

    def _maybe_schedule(
        self, name: str, param: Union[float, Callable[[mx.array], mx.array]]
    ):
        """
        To be used by derived classes to optionally put a parameter on a schedule.
        """
        if isinstance(param, Callable):
            self._schedulers[name] = param
            param = param(self.step)
        else:
            param = mx.array(param)
        self.state[name] = param


class SGD(Optimizer):
    r"""The stochastic gradient descent optimizer.

    Updates a parameter :math:`w` with a gradient :math:`g` as follows

    .. math::

        v_{t+1} &= \mu v_t + (1 - \tau) g_t \\
        w_{t+1} &= w_t - \lambda v_{t+1}

    Args:
        learning_rate (float or callable): The learning rate :math:`\lambda`.
        momentum (float, optional): The momentum strength :math:`\mu`. Default: ``0``
        weight_decay (float, optional): The weight decay (L2 penalty). Default: ``0``
        dampening (float, optional): Dampening for momentum :math:`\tau`. Default: ``0``
        nesterov (bool, optional): Enables Nesterov momentum. Default: ``False``
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening."
            )
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["v"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the SGD parameter update and stores :math:`v` in the
        optimizer state."""

        if self.weight_decay != 0:
            gradient += self.weight_decay * parameter

        if self.momentum <= 0:
            return parameter - self.learning_rate.astype(gradient.dtype) * gradient

        v = self.momentum * state.get("v")
        if self.dampening > 0:
            v += (1 - self.dampening) * gradient
        else:
            v += gradient

        if self.nesterov:
            update = gradient + self.momentum * v
        else:
            update = v

        state["v"] = v
        return parameter - self.learning_rate.astype(gradient.dtype) * update


class RMSprop(Optimizer):
    r"""The RMSprop optimizer [1].

    [1]: Tieleman, T. and Hinton, G. 2012. Lecture 6.5-rmsprop, coursera: Neural networks for machine learning

    .. math::

        v_{t+1} &= \alpha v_t + (1 - \alpha) g_t^2 \\
        w_{t+1} &= w_t - \lambda \frac{g_t}{\sqrt{v_{t+1}} + \epsilon}

    Args:
        learning_rate (float or callable): The learning rate :math:`\lambda`.
        alpha (float, optional): The smoothing constant :math:`\alpha`.
          Default: ``0.99``
        eps (float, optional): The term :math:`\epsilon` added to the denominator
          to improve numerical stability. Default: ``1e-8``
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        alpha: float = 0.99,
        eps: float = 1e-8,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.alpha = alpha
        self.eps = eps

        if self.alpha < 0.0:
            raise ValueError(
                f"RMSprop alpha should be >=0, {self.alpha} was provided instead"
            )
        if self.eps < 0.0:
            raise ValueError(
                f"RMSprop epsilon should be >0, {self.eps} was provided instead"
            )

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["v"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the RMSprop parameter update and stores :math:`v` in the optimizer state."""
        lr = self.learning_rate.astype(gradient.dtype)
        alpha = self.alpha
        eps = self.eps

        v = state["v"]
        v = alpha * v + (1 - alpha) * mx.square(gradient)
        state["v"] = v

        return parameter - lr * gradient / (mx.sqrt(v) + eps)


class Adagrad(Optimizer):
    r"""The Adagrad optimizer [1].

    Our Adagrad implementation follows the original paper. In detail,

    [1]: Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient methods
    for online learning and stochastic optimization. JMLR 2011.

    .. math::

        v_{t+1} &= v_t + g_t^2 \\
        w_{t+1} &= w_t - \lambda \frac{g_t}{\sqrt{v_{t+1}} + \epsilon}

    Args:
        learning_rate (float or callable): The learning rate :math:`\lambda`.
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-8``
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        eps: float = 1e-8,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.eps = eps

        if self.eps < 0.0:
            raise ValueError(
                f"Adagrad epsilon should be >0, {self.eps} was provided instead"
            )

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["v"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the Adagrad parameter update and stores :math:`v` in the
        optimizer state."""
        lr = self.learning_rate.astype(gradient.dtype)
        eps = self.eps

        v = state["v"] + mx.square(gradient)
        state["v"] = v

        return parameter - lr * gradient / (mx.sqrt(v) + eps)


class AdaDelta(Optimizer):
    r"""The AdaDelta optimizer with a learning rate [1].

    Our AdaDelta implementation follows the original paper. In detail,

    [1]: Zeiler, M.D., 2012. ADADELTA: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.

    .. math::

        v_{t+1} &= \rho v_t + (1 - \rho) g_t^2 \\
        \Delta w_{t+1} &= \frac{\sqrt{u_t + \epsilon}}{\sqrt{v_{t+1} + \epsilon}} g_t \\
        u_{t+1} &= \rho u_t + (1 - \rho) \Delta w_{t+1}^2 \\
        w_{t+1} &= w_t - \lambda \Delta w_{t+1}

    Args:
        learning_rate (float or callable): The learning rate :math:`\lambda`.
        rho (float, optional): The coefficient :math:`\rho` used for computing a
            running average of squared gradients. Default: ``0.9``
        eps (float, optional): The term :math:`\epsilon` added to the denominator to improve
          numerical stability. Default: `1e-8`
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        rho: float = 0.9,
        eps: float = 1e-6,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.rho = rho
        self.eps = eps
        if self.rho < 0.0:
            raise ValueError(
                f"AdaDelta rho should be >=0, {self.rho} was provided instead"
            )
        if self.eps < 0.0:
            raise ValueError(
                f"AdaDelta epsilon should be >0, {self.eps} was provided instead"
            )

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["v"] = mx.zeros_like(parameter)
        state["u"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the AdaDelta parameter update and stores :math:`v` and
        :math:`u` in the optimizer state."""
        lr = self.learning_rate.astype(gradient.dtype)
        rho = self.rho
        eps = self.eps

        v = state["v"]
        u = state["u"]

        v = rho * v + (1 - rho) * mx.square(gradient)
        d = mx.sqrt(u + eps) / mx.sqrt(v + eps) * gradient
        u = rho * u + (1 - rho) * mx.square(d)

        state["v"] = v
        state["u"] = u

        return parameter - lr * d


class Adam(Optimizer):
    r"""The Adam optimizer [1]. In detail,

    [1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic
    optimization. ICLR 2015.

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
        w_{t+1} &= w_t - \lambda \frac{m_{t+1}}{\sqrt{v_{t+1} + \epsilon}}

    Args:
        learning_rate (float or callable): The learning rate :math:`\lambda`.
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing running averages of the
          gradient and its square. Default: ``(0.9, 0.999)``
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-8``
        bias_correction (bool, optional): If set to ``True``, bias correction
          is applied. Default: ``False``
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-8,
        bias_correction: bool = False,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.eps = eps
        self.bias_correction = bias_correction

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the Adam parameter update and stores :math:`v` and
        :math:`m` in the optimizer state."""
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        eps = self.eps
        bias_correction = self.bias_correction
        step = self.step

        m = state["m"]
        v = state["v"]
        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * mx.square(gradient)
        state["m"] = m
        state["v"] = v

        if bias_correction:
            numerator = lr / (1 - b1**step) * m
            denominator = mx.sqrt(v) / mx.sqrt(1 - b2**step) + eps
            return parameter - numerator / denominator
        else:
            return parameter - lr * m / (mx.sqrt(v) + eps)


class AdamW(Adam):
    r"""The AdamW optimizer [1]. We update the weights with a weight_decay
    (:math:`\lambda`) value:

    [1]: Loshchilov, I. and Hutter, F., 2019. Decoupled weight decay
    regularization. ICLR 2019.

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
        w_{t+1} &= w_t - \alpha (\frac{m_{t+1}}{\sqrt{v_{t+1} + \epsilon}} + \lambda w_t)

    Args:
        learning_rate (float or callable): The learning rate :math:`\alpha`.
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing running averages of the
          gradient and its square. Default: ``(0.9, 0.999)``
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-8``
        weight_decay (float, optional): The weight decay :math:`\lambda`.
          Default: ``0``.
        bias_correction (bool, optional): If set to ``True``, bias correction
          is applied. Default: ``False``
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        bias_correction: bool = False,
    ):
        super().__init__(
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
            bias_correction=bias_correction,
        )
        self.weight_decay = weight_decay

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the AdamW parameter update by modifying the parameters
        passed into Adam.
        """

        lr = self.learning_rate.astype(gradient.dtype)
        return super().apply_single(
            gradient, parameter * (1 - lr * self.weight_decay), state
        )


class Adamax(Adam):
    r"""The Adamax optimizer, a variant of Adam based on the infinity norm [1].

    Our Adam implementation follows the original paper and omits the bias
    correction in the first and second moment estimates. In detail,

    [1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic
    optimization. ICLR 2015.

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \max(\beta_2 v_t, |g_t|) \\
        w_{t+1} &= w_t - \lambda \frac{m_{t+1}}{v_{t+1} + \epsilon}

    Args:
        learning_rate (float or callable): The learning rate :math:`\lambda`.
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing running averages of the
          gradient and its square. Default: ``(0.9, 0.999)``
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-8``
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-8,
    ):
        super().__init__(learning_rate, betas, eps)
        if not 0.0 <= eps:
            raise ValueError(
                f"Epsilon value should be >=0, {self.eps} was provided instead"
            )

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the Adamax parameter update and stores :math:`v` and
        :math:`m` in the optimizer state."""
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        eps = self.eps

        m = state["m"]
        v = state["v"]

        m = b1 * m + (1 - b1) * gradient
        v = mx.maximum(b2 * v, mx.abs(gradient))
        state["m"] = m
        state["v"] = v

        return parameter - lr * m / (v + eps)


class Lion(Optimizer):
    r"""The Lion optimizer [1].

    Since updates are computed through the sign operation, they tend to
    have larger norm than for other optimizers such as SGD and Adam.
    We recommend a learning rate that is 3-10x smaller than AdamW and a
    weight decay 3-10x larger than AdamW to maintain the strength
    (lr * wd). Our Lion implementation follows the original paper. In
    detail,

    [1]: Chen, X. Symbolic Discovery of Optimization Algorithms. arXiv
    preprint arXiv:2302.06675.

    .. math::

        c_{t + 1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        m_{t + 1} &= \beta_2 m_t + (1 - \beta_2) g_t \\
        w_{t + 1} &= w_t - \eta (\text{sign}(c_t) + \lambda w_t)

    Args:
        learning_rate (float or callable): The learning rate :math:`\eta`.
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing the gradient
          momentum and update direction. Default: ``(0.9, 0.99)``
        weight_decay (float, optional): The weight decay :math:`\lambda`. Default: ``0.0``
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        betas: List[float] = [0.9, 0.99],
        weight_decay: float = 0.0,
    ):
        super().__init__()

        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = betas
        self.weight_decay = weight_decay

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["m"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the Lion parameter update and stores :math:`m`
        in the optimizer state."""
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        weight_decay = self.weight_decay

        m = state["m"]
        c = b1 * m + (1 - b1) * gradient
        state["m"] = b2 * m + (1 - b2) * gradient
        if weight_decay > 0:
            parameter = (1 - lr * weight_decay) * parameter
        return parameter - lr * mx.sign(c)


class Adafactor(Optimizer):
    r"""The Adafactor optimizer.

    Our Adafactor implementation follows the original paper: `Adafactor:
    Adaptive Learning Rates with Sublinear Memory Cost
    <https://arxiv.org/abs/1804.04235>`_

    Args:
        learning_rate (float or callable, optional): The learning rate.
            Default: ``None``.
        eps (tuple(float, float), optional): The first term :math:`\epsilon_1`
            added to the square of the gradients to improve numerical
            stability and the second term :math:`\epsilon_2` is used for
            parameter scaling if ``parameter_scale`` is set to ``True``.
            Default: ``(1e-30, 1e-3)``.
        clip_threshold (float, optional): Clips the unscaled update at
            ``clip_threshold``. Default: ``1.0``.
        decay_rate (float, optional): Coefficient for the running average
            of the squared gradient. Default: ``-0.8``.
        beta_1 (float, optional): If set to a value bigger than zero
            then first moment will be used. Default: ``None``.
        weight_decay (float, optional): The weight decay :math:`\lambda`.
            Default: ``0.0``.
        scale_parameter (bool, optional): If set to ``True`` the learning rate
            will be scaled by :math:`\max(\epsilon_1, \text{RMS}(w_{t-1}))`.
            Default: ``True``.
        relative_step (bool, optional): If set to ``True`` the ``learning_rate``
            will be ignored and relative step size will be computed.
            Default: ``True``.
        warmup_init (bool, optional): If set to ``True`` then the relative
            step size will be calculated by the current step. Default:
            ``False``.
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array], None] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta_1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
    ):
        super().__init__()
        if learning_rate is not None:
            self._maybe_schedule("learning_rate", learning_rate)
        self.eps = eps
        self.clip_threshold = clip_threshold
        self.decay_rate = decay_rate
        self.beta_1 = beta_1
        self.weight_decay = weight_decay
        self.scale_parameter = scale_parameter
        self.relative_step = relative_step
        self.warmup_init = warmup_init

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        if parameter.ndim >= 2:
            shape = parameter.shape
            dtype = parameter.dtype
            state["exp_avg_sq_row"] = mx.zeros(shape[:-1], dtype=dtype)
            state["exp_avg_sq_col"] = mx.zeros(shape[:-2] + shape[-1:], dtype=dtype)
        else:
            state["exp_avg_sq"] = mx.zeros_like(parameter)

        if self.beta_1 is not None:
            state["exp_avg"] = mx.zeros_like(parameter)

    def _compute_rms(self, inputs):
        return mx.sqrt(mx.mean(mx.square(inputs)))

    def _compute_learning_rate(self, step, parameter_rms):
        if self.relative_step:
            min_step = 1e-6 * step if self.warmup_init else 1e-2
            relative_step_size = mx.minimum(min_step, mx.rsqrt(step))
        else:
            relative_step_size = self.learning_rate

        relative_step_size = relative_step_size.astype(parameter_rms.dtype)
        parameter_scale = 1.0
        if self.scale_parameter:
            parameter_scale = mx.maximum(self.eps[1], parameter_rms)
        return parameter_scale * relative_step_size

    def _approximate_exp_moving_avg(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = mx.rsqrt(
            exp_avg_sq_row / mx.mean(exp_avg_sq_row, axis=-1, keepdims=True)
        )
        c_factor = mx.rsqrt(exp_avg_sq_col)
        return mx.matmul(
            mx.expand_dims(r_factor, axis=-1), mx.expand_dims(c_factor, axis=0)
        )

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the Adafactor parameter and state update."""
        factored = gradient.ndim >= 2

        step = self.step
        use_first_moment = self.beta_1 is not None

        parameter_rms = self._compute_rms(parameter)
        learning_rate = self._compute_learning_rate(step, parameter_rms)
        beta_2 = 1.0 - (step**self.decay_rate).astype(parameter_rms.dtype)
        update = mx.square(gradient) + self.eps[0]

        if factored:
            exp_avg_sq_row = state["exp_avg_sq_row"]
            exp_avg_sq_col = state["exp_avg_sq_col"]
            exp_avg_sq_row = (beta_2 * exp_avg_sq_row) + (
                (1 - beta_2) * mx.mean(update, axis=-1)
            )
            exp_avg_sq_col = (beta_2 * exp_avg_sq_col) + (
                (1 - beta_2) * mx.mean(update, axis=-2)
            )
            state["exp_avg_sq_row"] = exp_avg_sq_row
            state["exp_avg_sq_col"] = exp_avg_sq_col
            update = self._approximate_exp_moving_avg(exp_avg_sq_row, exp_avg_sq_col)
            update = update * gradient
        else:
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq = (beta_2 * exp_avg_sq) + ((1 - beta_2) * update)
            state["exp_avg_sq"] = exp_avg_sq
            update = mx.rsqrt(exp_avg_sq) * gradient

        update = update / mx.maximum(
            1.0, self._compute_rms(update) / self.clip_threshold
        )
        update = learning_rate * update

        if use_first_moment:
            exp_avg = state["exp_avg"]
            exp_avg = (self.beta_1 * exp_avg) + ((1 - self.beta_1) * update)
            state["exp_avg"] = exp_avg
            update = exp_avg

        if self.weight_decay != 0:
            parameter += parameter * (-self.weight_decay * learning_rate)
        return parameter - update


class Muon(Optimizer):
    r"""The Muon optimizer - MomentUm Orthogonalized by Newton-schulz.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, a Newton-Schulz iteration is used, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    For more details, see: https://kellerjordan.github.io/posts/muon/

    Note:
        - This optimizer may not be optimal for the embedding layer, the final fully connected layer,
          or any 0D/1D parameters; those should be optimized by a standard method (e.g., AdamW).
        - For 4D convolutional filters, it works by flattening their last dimensions.

    Args:
        learning_rate (float or callable): The learning rate used by the internal SGD.
        momentum (float, optional): The momentum strength. Default: ``0.95``
        weight_decay (float, optional): The weight decay (L2 penalty). Default: ``0.01``
        nesterov (bool, optional): Enables Nesterov momentum. Recommended for better performance. 
            Default: ``True``
        ns_steps (int, optional): Number of Newton-Schulz iteration steps for orthogonalization. 
            Default: ``5``
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]],
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        super().__init__()
        
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.ns_steps = ns_steps

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize optimizer state"""
        state["v"] = mx.zeros_like(parameter)

    def _zeropower_via_newtonschulz5(self, G, steps: int):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
        quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
        of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
        zero even beyond the point where the iteration no longer converges all the way to one everywhere
        on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
        where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
        performance at all relative to UV^T, where USV^T = G is the SVD.
        """
        assert G.ndim >= 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.astype(mx.bfloat16)
        transpose_needed = G.shape[-2] > G.shape[-1]
        
        if transpose_needed:
            X = X.T
        
        # Ensure spectral norm is at most 1
        norm = mx.sqrt(mx.sum(X * X, axis=(-2, -1), keepdims=True) + 1e-7)
        X = X / norm
        
        # Perform the NS iterations
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        
        if transpose_needed:
            X = X.T
        return X

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Performs the Muon parameter update"""
        
        # Apply weight decay
        if self.weight_decay != 0:
            gradient = gradient + self.weight_decay * parameter
        
        # Update momentum buffer
        v = self.momentum * state.get("v")
        v = v + (1 - self.momentum) * gradient
        state["v"] = v
        
        # Get effective gradient
        if self.nesterov:
            effective_grad = gradient * self.momentum + v * (1 - self.momentum)
        else:
            effective_grad = v
        
        # For tensors with fewer than 2 dimensions, skip Newton-Schulz
        if effective_grad.ndim < 2:
            orthogonalized_grad = effective_grad
            scale_factor = 1.0
        else:
            # Save original shape for 4D conv filters
            original_shape = effective_grad.shape
            reshape_needed = effective_grad.ndim > 2
            
            if reshape_needed:
                effective_grad = mx.reshape(effective_grad, (effective_grad.shape[0], -1))
            
            # Apply Newton-Schulz orthogonalization
            orthogonalized_grad = self._zeropower_via_newtonschulz5(effective_grad, steps=self.ns_steps)
            
            # Reshape back if needed
            if reshape_needed:
                orthogonalized_grad = mx.reshape(orthogonalized_grad, original_shape)
            
            # Calculate scaling factor
            scale_factor = max(1, parameter.shape[-2] / parameter.shape[-1]) ** 0.5
        
        return parameter - self.learning_rate.astype(gradient.dtype) * orthogonalized_grad * scale_factor


def clip_grad_norm(grads, max_norm):
    """Clips the global norm of the gradients.

    This function ensures that the global norm of the gradients does not exceed
    ``max_norm``. It scales down the gradients proportionally if their norm is
    greater than ``max_norm``.

    Example:
        >>> grads = {"w1": mx.array([2, 3]), "w2": mx.array([1])}
        >>> clipped_grads, total_norm = clip_grad_norm(grads, max_norm=2.0)
        >>> print(clipped_grads)
        {"w1": mx.array([...]), "w2": mx.array([...])}

    Args:
        grads (dict): A dictionary containing the gradient arrays.
        max_norm (float): The maximum allowed global norm of the gradients.

    Returns:
        (dict, float): The possibly rescaled gradients and the original
        gradient norm.
    """
    norm_squared = tree_reduce(lambda acc, g: acc + g.square().sum(), grads, 0.0)
    total_norm = mx.sqrt(norm_squared)
    normalizer = max_norm / (total_norm + 1e-6)

    def clipper(g):
        return mx.where(total_norm < max_norm, g, g * normalizer)

    clipped_grads = tree_map(clipper, grads)
    return clipped_grads, total_norm
