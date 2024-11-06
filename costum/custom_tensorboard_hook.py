from typing import Optional

from mmcv.runner import HOOKS, Hook
from torch.utils.tensorboard import SummaryWriter
from mmcv.runner.hooks.logger.base import LoggerHook



@HOOKS.register_module()
class TensorboardCustomMetricsHook(LoggerHook):
    def __init__(self,
                 log_dir: Optional[str] = None,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.writer = SummaryWriter(log_dir)
        self.interval = interval


    def after_train_iter(self, runner):
        # 定义自定义指标的记录逻辑
        if runner.iter % self.interval == 0:
            # 遍历所有自定义指标
            for metric in self.custom_metrics:
                if metric in runner.log_buffer.output:
                    value = runner.log_buffer.output[metric]
                    self.writer.add_scalar(f'CustomMetrics/{metric}', value, runner.iter)


    def after_val_epoch(self, runner):
        # 记录验证阶段的自定义指标
        for metric in self.custom_metrics:
            if metric in runner.log_buffer.output:
                value = runner.log_buffer.output[metric]
                self.writer.add_scalar(f'CustomMetrics/{metric}', value, runner.epoch)


    def after_run(self, runner):
        # 训练结束时关闭 writer
        self.writer.close()