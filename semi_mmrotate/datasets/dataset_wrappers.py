from mmdet.datasets import DATASETS, ConcatDataset, build_dataset

@DATASETS.register_module()
class SemiDataset(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup: dict, unsup: dict, **kwargs):
        super().__init__([build_dataset(sup), build_dataset(unsup)], **kwargs)
    
    # 下面这两个函数似乎用不到
    @property
    def sup(self):
        return self.datasets[0]

    @property
    def unsup(self):
        return self.datasets[1]





@DATASETS.register_module()
class SparseDataset(ConcatDataset):
    """Wrapper for sparsesupervised od."""

    def __init__(self, unsup: dict, **kwargs):
        super().__init__([build_dataset(unsup)], **kwargs)

    # 下面这个函数似乎用不到
    @property
    def unsup(self):
        return self.datasets[0]
