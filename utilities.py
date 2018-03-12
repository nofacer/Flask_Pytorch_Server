import torchvision.transforms as transforms
import PIL


def array2tensor(img):
    img = PIL.Image.fromarray(img.astype('uint8'))
    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    img = trans(img)
    return img
