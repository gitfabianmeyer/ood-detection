from clearml import Task, Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

task = Task.init(project_name="ma_fmeyer", task_name="FIRST-STEPS")
dataset_name = "MNIST-Dataset"
dataset_path = Dataset.get(
    dataset_project='MNIST',
    dataset_name=dataset_name).get_local_copy()

print(dataset_path)
dset = MNIST(root=dataset_path)
dloader = DataLoader(dset)

for i, (imgs, targes) in enumerate(dloader):
    if i % 50 == 0:
        print(f"Still running at {i} / {len(dset)}")