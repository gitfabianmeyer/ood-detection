import os

import torch
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from clearml import Dataset, Task

task = Task.init(project_name="ma_fmeyer", task_name="FIRST-STEPS")
dataset_name = "COCO 2017 Dataset"
DATASET_PATH = Dataset.get(
    dataset_project='COCO-2017',
    dataset_name=dataset_name
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
print(DATASET_PATH)


class MyCocoDetection:
    def __init__(self, train=True):
        if train:
            filename = 'train2017'
        else:
            filename = 'val2017'
            print('file is val')

        super(MyCocoDetection, self).__init__()

        self.transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),  # 224 for vit, 288 for res50x4
            CenterCrop(224),  # 224 for vit, 288 for res50x4
            ToTensor(),
            Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        ])

        self.coco_dataset = CocoDetection(
            root=os.path.join(f'{DATASET_PATH}/images', filename),
            annFile=os.path.join('{}/annotations'.format(DATASET_PATH),
                                 'captions_{}.json'.format(filename)))

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, index):
        img = self.transform(self.coco_dataset[index][0])
        captions = self.coco_dataset[index][1]
        cap_list = []
        for i, caption in enumerate(captions):
            if i == 5:
                # print('more than 5 captions for this image', index)
                break
            cap = caption['caption']
            cap_list.append(cap)
        if len(cap_list) < 5:
            print('has less than 5 captions', index)
        return img, cap_list


if __name__ == '__main__':
    task.execute_remotely('5e62040adb57476ea12e8593fa612186')

    dset = MyCocoDetection(train='False')
    print(len(dset))
    dloader = DataLoader(dset)
    i = 0
    for data, target in tqdm(dloader):
        i+=1
        if i % 1000 == 0:
            print(f"At {i} and still running")
