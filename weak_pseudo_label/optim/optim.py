import torch
from .build import OPTIM

@OPTIM.register_module()
class SGDPolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        param_groups = [
            dict(params = params[0], lr = lr, weight_decay = weight_decay),
            dict(params = params[1], lr = 2 * lr, weight_decay = 0),
            dict(params = params[2], lr = 10 * lr, weight_decay = weight_decay),
            dict(params = params[3], lr = 20 * lr, weight_decay = 0),
        ]
        super().__init__(param_groups, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


@OPTIM.register_module()
class AdamPolyOptimizer(torch.optim.Adam):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        param_groups = [
            dict(params = params[0], lr = lr, weight_decay = weight_decay),
            dict(params = params[1], lr = 2 * lr, weight_decay = 0),
            dict(params = params[2], lr = 10 * lr, weight_decay = weight_decay),
            dict(params = params[3], lr = 20 * lr, weight_decay = 0),
        ]
        super().__init__(param_groups, lr, weight_decay= weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        super().step(closure)
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult
        self.global_step += 1