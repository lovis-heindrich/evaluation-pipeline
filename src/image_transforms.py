import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageDraw
import math

def noise_effect(sample, cutoff, factor):
    #noisefilter = torch.rand(1,64,64)
    #noisefilter[noisefilter<cutoff] = 0
    #noisefilter = torch.cat([noisefilter]*3)
    noisefilter = torch.rand(3,64,64)
    sample = sample*(1-factor) + noisefilter*factor
    sample = torch.clamp(sample, 0, 1)
    return sample

def gen_noise_transform(scale):
    return transforms.Compose(
    [transforms.ToPILImage(),   
     transforms.ToTensor(),
     transforms.Lambda(lambda img: noise_effect(img, 0.5, scale))
     ])

def apply_greyscale(img, intensity):
    grey_img = grey_mask_transforms(img)
    img = img*(1-intensity) + grey_img*intensity
    return torch.clamp(img, 0, 1)

grey_mask_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(3),
    transforms.ToTensor()])

def gen_grey_transform(scale):
    return transforms.Compose(
    [transforms.ToPILImage(),   
     #transforms.Grayscale(3),
     #transforms.ColorJitter(brightness=1.4, contrast=0, saturation=0, hue=0),
     transforms.ToTensor(),
     transforms.Lambda(lambda img: apply_greyscale(img, scale)),
     ])

def snow_effect(sample, intensities, snow_img):
    for intensity in intensities:
        noise = mask_transforms_snow(snow_img) 
        sample = torch.clamp(sample + noise*intensity, 0, 1)
    return sample

mask_transforms_snow = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop((64,64), (0.01, 0.02)),
    transforms.ColorJitter(brightness=(1.2,1.3), contrast=0, saturation=0, hue=0),
    transforms.ToTensor()]
)

def gen_snow_transform(intensities, snow_image):
    return transforms.Compose(
    [transforms.ToPILImage(), 
     transforms.ColorJitter(brightness=(0.8, 0.8), contrast=0, saturation=0, hue=0),
     transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius = 1))),
     transforms.ToTensor(),
     transforms.Lambda(lambda img: snow_effect(img, intensities, snow_image))
     ])

def rain_effect(sample, intensities, degree, rain_image):
    degree = torch.rand((1))*2*degree - degree
    mask_transforms_rain = gen_rain_mask_transforms(degree)
    for intensity in intensities:
        noise = mask_transforms_rain(rain_image)
        #imshow(noise)
        sample = sample + intensity * noise
        #sample[(noise>0.5) & (sample>0.9)] = 0.9
        sample = torch.clamp(sample, 0, 1)
    return sample

def gen_rain_mask_transforms(degree):
    mask_transforms = transforms.Compose([
        transforms.RandomRotation((degree, degree+2)),
        transforms.RandomResizedCrop((64,64), (0.01, 0.02)),
        transforms.ColorJitter(brightness=(1,2), contrast=0, saturation=0, hue=0),
        #transforms.RandomCrop((64,64)),
        transforms.ToTensor()])
    return mask_transforms

def gen_rain_transforms(intensities, degree, rain_image):
    return transforms.Compose(
    [transforms.ToPILImage(), 
     transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius = 1))),
     transforms.ToTensor(),
     transforms.Lambda(lambda img: rain_effect(img, intensities, degree, rain_image))
     ])

def add_fog(img, scale):
    noisefilter = torch.rand(1,64,64)
    noisefilter = torch.cat([noisefilter]*3)
    noisefilter = mask_transforms_fog(noisefilter)
    img = img*(1-scale) + scale*noisefilter
    img = torch.clamp(img, 0, 1)
    return img

mask_transforms_fog = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius = 4))),
    transforms.ToTensor()
])

def gen_fog_transform(scale):
    return transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ColorJitter(brightness=(0.7, 0.8), contrast=0, saturation=0, hue=0),
     transforms.ToTensor(),
     transforms.Lambda(lambda img: add_fog(img, scale))
     ])

def gen_perspective_transform(distortion, croppercent):
    return transforms.Compose(
    [transforms.ToPILImage(), 
     transforms.RandomPerspective(distortion, 1),
     transforms.CenterCrop(math.floor(croppercent*64)),
     transforms.Resize(64),
     transforms.ToTensor(),
     ])

def gen_blur_transform(size):
    return transforms.Compose(
    [transforms.ToPILImage(), 
     transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius = size))),
     transforms.ToTensor(),
     ])
    
def gen_crop_transform(crop_size, original_size=64):
    return transforms.Compose(
    [transforms.ToPILImage(), 
     transforms.RandomCrop((crop_size, crop_size)),
     transforms.Resize((original_size, original_size)),  
     transforms.ToTensor()
     ])

def gen_rotate_transform(rot):
    return transforms.Compose(
    [transforms.ToPILImage(),
     transforms.RandomRotation(rot),
     transforms.ToTensor()
     ])

def add_reflection(img, size, scale):
    w, h = 64, 64

    pos = torch.randint(low=0, high=64-size, size=(2,1))
    x, y = pos[0].item(), pos[1].item()
    shape = [(x, y), (x+size, y+size)] 
  
    # creating new Image object 
    rect = Image.new("RGB", (w, h)) 
  
    # create rectangle image 
    draw = ImageDraw.Draw(rect)   
    draw.rectangle(shape, fill ="#ffffff") 
    rect = reflection_mask_transform(rect)
    img = torch.clamp(img + scale*rect, 0, 1)
    return img


reflection_mask_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius = 5))),                   
    transforms.ToTensor()
])

def gen_reflection_transform(size, scale):
    return transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
     transforms.ToTensor(),
     transforms.Lambda(lambda img: add_reflection(img, size, scale))
     ])