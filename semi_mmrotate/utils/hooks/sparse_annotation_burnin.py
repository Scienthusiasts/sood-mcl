from mmcv.parallel import is_module_wrapper
from mmcv.runner import Hook
from mmcv.runner.hooks import HOOKS

@HOOKS.register_module()
class SparseAnnotationBurnInHook(Hook):
    """
    Hook to manage a burn-in phase for sparse annotation training.
    Args:
        start_step (int): The step at which burn-in starts.
        end_step (int): The step at which burn-in ends.
    """
    def __init__(self, end_step=1000):
        self.end_step = end_step

    def before_train_iter(self, runner):
        cur_iter = runner.iter

        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if cur_iter < self.end_step:
            model.burn_in_mode = True
        else:
            model.burn_in_mode = False