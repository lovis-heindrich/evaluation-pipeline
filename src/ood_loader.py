import torch
from torch.utils.data import TensorDataset, Dataset
from torchvision import transforms
import torchvision
import pickle
import numpy as np
import skimage.transform
import gan

# Dataset wrapper for OOD detection that overwrites the original label
class OOD(Dataset):
    def __init__(self, original_dataset, label, transform = None):
        self.original_dataset = original_dataset
        self.label = label
        self.transform = transform
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, i: int):
        image, original_label = self.original_dataset[i]
        label = self.label[i]
        if self.transform is not None:
            image = self.transform(image)
        return (image, label)

# Dataset that mixes two images of different labels
class MIX_CURE_TSD(Dataset):
    def __init__(self, dataset, transform=None):
        self.images = dataset.images
        # Noise as OOD data - label 0
        self.labels = torch.zeros(len(self.images))
        self.original_labels = dataset.labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int):
        image = np.copy(self.images[idx])

        # indices of images of a different class
        indices = np.where(self.original_labels != self.original_labels[idx])[0]
        selection = np.random.choice(indices)
        image2 = self.images[selection]
        
        image[:,:64,:32] = image2[:,:64,:32]
        image = torch.from_numpy(image).float()
        if self.transform is not None:
           return self.transform(image), self.labels[idx]
        else:
            return (image, self.labels[idx])

class OodLoader():
    def __init__(self, num_channels, image_size, ood_label, device):
        self.num_channels = num_channels
        self.image_size = image_size
        self.ood_label = ood_label
        self.device = device
        self.oodlen = len(ood_label)

    def get_noise_dataset(self, uniform):
        if uniform:
            noise = torch.rand(self.oodlen, self.num_channels, self.image_size, self.image_size)
        else:
            noise = torch.randn(self.oodlen, self.num_channels, self.image_size, self.image_size)
            noise = torch.clamp(noise, 0, 1)
        return TensorDataset(noise, self.ood_label)

    def get_tiny_imagenet_dataset(self, path):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.ToTensor(),])
        imgs = torchvision.datasets.ImageFolder(path, transform)
        return OOD(imgs, self.ood_label)

    def load_ood_from_disk(self, path, normalize=True):
        with open(path, 'rb') as f:
            examples = pickle.load(f)
            # Range 0-1
            if normalize:
                examples = examples / 2 + 0.5
        return examples

    def get_stored_tensor_dataset(self, path, normalize=False):
        images = self.load_ood_from_disk(path)
        data = TensorDataset(images[:self.oodlen], self.ood_label)
        return data

    def shuffle(self, image):
        # Clone to avoid in-place edit
        image = image.clone().numpy()
        image = np.transpose(image, (1,2,0))
        shape = image.shape
        image = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
        # Shuffles array along the first axis - RGB channels stay intact
        np.random.shuffle(image)
        image = np.reshape(image, shape)
        # Reconversion done by ToTensor
        #image_final = np.transpose(image, (2,0,1))
        return image

    def get_shuffle_dataset(self, dataset):
        transform = transforms.Compose([
            transforms.Lambda(lambda img: self.shuffle(img)),
            transforms.ToTensor()])
    
        return OOD(dataset, self.ood_label, transform)
    
    def get_mix_dataset(self, dataset):
        return MIX_CURE_TSD(dataset)

    def swirl(self, image):
        # Clone to avoid in-place edit
        image = image.clone().numpy()
        image = np.transpose(image, (1,2,0))
        image = skimage.transform.swirl(image, center=None, strength=3, radius=100, rotation=0)
        return image

    def get_swirl_dataset(self, dataset):
        transform = transforms.Compose([
            transforms.Lambda(lambda img: self.swirl(img)),
            transforms.ToTensor(),
        ])

        return OOD(dataset, self.ood_label, transform)
    
    def color_shift(self, image):
        image = image.clone()
        channels = image.shape[0]
        image = [image[(i+1)%channels,:,:] for i in range(channels)]
        image = torch.stack(image)
        return image
    
    def get_color_shift_dataset(self, dataset):
        transform = transforms.Compose([
            transforms.Lambda(lambda img: self.color_shift(img))
        ])

        return OOD(dataset, self.ood_label, transform)
    
    def get_gan_dataset(self, path, normalize=True):
        with torch.no_grad():
            save_res = torch.load(path)
            params = save_res["params"]
            generator = gan.initGenerator(noise_size=params["noise_size"], num_channels=params["num_channels"], conditional=params["conditional"], lee=params["lee"], wgan=params["wgan"])
            generator.load_state_dict(save_res["generator"])
            generator.to(self.device)
            if save_res["params"]["conditional"]:
                x = generator(torch.randn(self.oodlen, params["noise_size"], 1, 1).to(self.device), torch.randint(params["num_classes"], [self.oodlen]).to(self.device)).to("cpu")
            else:
                x = generator(torch.randn(self.oodlen, params["noise_size"], 1, 1).to(self.device), torch.randint(params["num_classes"], [self.oodlen]).to(self.device)).to("cpu")
            if normalize:
                x = x / 2 + 0.5
            #utils.gridshow(x[0:32,:,:,:])
            examples = x.detach()
            sricharan = TensorDataset(examples, self.ood_label)
            del generator, save_res, x
            return sricharan
        
    def get_gan(self, path):
        save_res = torch.load(path)
        with torch.no_grad():
            return GAN_wrapper(save_res["generator"], save_res["params"], self.device, self.oodlen, self.ood_label)

class GAN_wrapper():
    def __init__(self, state_dict, params, device, oodlen, ood_label):
        self.noise_size = params["noise_size"]
        self.num_channels=params["num_channels"]
        self.conditional=params["conditional"]
        self.lee=params["lee"]
        self.wgan=params["wgan"]
        self.num_classes = params["num_classes"]
        self.device = device
        self.oodlen = oodlen
        self.ood_label = ood_label

        self.generator = gan.initGenerator(noise_size=self.noise_size, num_channels=self.num_channels, conditional=self.conditional, lee=self.lee, wgan=self.wgan)
        self.generator.load_state_dict(state_dict)
        self.generator.to(self.device)

    def get_gan_examples(self, normalize=True):
        if self.conditional:
            x = self.generator(torch.randn(self.oodlen, self.noise_size, 1, 1).to(self.device), torch.randint(self.num_classes, [self.oodlen]).to(self.device)).to("cpu")
        else:
            x = self.generator(torch.randn(self.oodlen, self.noise_size, 1, 1).to(self.device)).to("cpu")
        if normalize:
            x = x / 2 + 0.5
        examples = x.detach()
        ds = TensorDataset(examples, self.ood_label)
        del x
        return ds

    def get_gan_images(self, n=16, normalize=True):
        if self.conditional:
            x = self.generator(torch.randn(n, self.noise_size, 1, 1).to(self.device), torch.randint(self.num_classes, [n]).to(self.device)).to("cpu")
        else:
            x = self.generator(torch.randn(n, self.noise_size, 1, 1).to(self.device)).to("cpu")
        if normalize:
            x = x / 2 + 0.5
        examples = x.detach()
        return examples