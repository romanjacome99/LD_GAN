import numpy as np


class SigmoidScheduler:
    def __init__(self, param_init, max_epoch, alpha=0.05, offset=0, param_min=1, inverse=True):
        self.param_min = param_min
        self.param_init = param_init
        self.max_epoch = max_epoch
        self.inverse = inverse
        self.alpha = alpha
        self.offset = offset
        center = offset + (max_epoch - offset) // 2
        self.name = f"SigmoidDecay, alpha {alpha}"

        # Initial sigmoid function

        fn = lambda x: (param_init - param_min) / (1 + np.exp(alpha * (x - center))) + param_min
        param_end = fn(max_epoch)
        self.fn = lambda x: (param_init - param_min) / (param_init - param_end) * (fn(x) - param_init) + param_init

    def step(self, epoch):
        if self.inverse:
            return self.fn(self.max_epoch - epoch)
        else:
            return self.fn(epoch)


class ExponentialScheduler:
    def __init__(self, param_init, max_epoch, alpha=0.05, param_min=1e-7, inverse=True):
        self.e_max = max_epoch
        self.alpha = float(alpha)
        self.a = (param_min - param_init) / (np.exp(-self.alpha * max_epoch) - 1)
        self.b = param_init - self.a
        self.inverse = inverse
        self.name = f"ExponentialDecay, alpha {alpha}"

        # Initial exponential function
        self.fn = lambda x: self.a * np.exp(-self.alpha * min(x, self.e_max)) + self.b

    def step(self, epoch):
        if self.inverse:
            return self.fn(self.e_max - epoch)
        else:
            return self.fn(epoch)
