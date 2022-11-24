import numpy
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class IsolatedClass(Dataset):
    def __init__(self, dataset, class_label=None):
        assert class_label, 'a semantic label must be specified'
        self.transform = dataset.transform
        self.class_label = class_label
        class_mask = np.array(dataset.targets) == dataset.class_to_idx[class_label]
        self.data = [dataset.data[i] for i in range(len(dataset.data)) if class_mask[i]]
        self.targets = np.array(dataset.targets[class_mask])

    def __getitem__(self, idx):
        image_file = self.data[idx]
        print()
        try:
            # works for most datasets
            img = Image.open(image_file)
        except AttributeError:
            try:
                if type(image_file) == numpy.ndarray:
                    img = Image.fromarray(image_file)
                else:
                    img = Image.fromarray(image_file.numpy(), mode="L")
            except TypeError:  # for svhn b/w images
                img = Image.fromarray(np.transpose(image_file, (1, 2, 0)))

        img = img.convert('RGB')
        return self.transform(img)
        # return self.transform(img.convert('RGB'))

    def __len__(self):
        return len(self.data)

    @property
    def name(self):
        return self.class_label


def single_isolated_class_loader(full_dataset, batch_size=1):
    loaders_dict = {}
    labels = full_dataset.classes
    for label in labels:
        dataset = IsolatedClass(full_dataset, label)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4)
        loaders_dict[label] = loader

    return loaders_dict


def get_loader(dataset):
    return DataLoader(dataset, batch_size=128)
