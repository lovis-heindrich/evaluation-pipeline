import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms

class CURE_TSD(Dataset):
    def __init__(self, path, transform=None):
        dataset = np.load(path)
        self.images = dataset['arr_0']
        self.labels = dataset['arr_1']
        self.labels -= 1 # Make first label 0
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int):
        image = self.images[idx]
        image = torch.from_numpy(image).float()
        if self.transform is not None:
            image = self.transform(image)
        return (image, self.labels[idx])

def get_dataloader(path, train_size, batch_size, num_workers=2, normalize=False):
    if normalize:
        transform = transforms.Compose(
        [transforms.Normalize((0.5,), (0.5,))])
        data = CURE_TSD(path, transform)
    else:
        data = CURE_TSD(path)

    if train_size < 1:
        train_size = int(train_size * len(data))
        test_size = len(data) - train_size
        trainset_raw, testset_raw = torch.utils.data.random_split(data, [train_size, test_size])
        trainset = trainset_raw
        testset = testset_raw
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, num_workers=0)
    else:
        trainloader = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        testloader = None
        trainset = data
        testset = None
    return trainloader, testloader, trainset, testset, data