from datasets.imagenet import OodTinyImageNet
from datasets.corruptions import Corruptions


class OodImagenetO(OodTinyImageNet):
    def __init__(self, data_path, transform, train, templates=None):
        super(OodImagenetO, self).__init__(data_path, transform, train, templates=templates)

    self.corruptions = Corruptions
    self.idx_to_corruption = {i: corr for (i, corr) in enumerate(self.corruptions.keys())}
