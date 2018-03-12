import PIL
import torch
from model import model
from shape_maker import Shape
from torch.autograd import Variable
import torchvision.transforms as transforms

# Load structure
net = model.ShapeDetectNetwork()
# Load para
net.load_state_dict(torch.load('./model/shapeDetect', map_location=lambda storage, loc: storage))
# Create two shapes
creator = Shape()
circle = creator.make('circle')
retrangle = creator.make('retrangle')


# transform img array to pytorch tensor
def array2tensor(img):
    img = PIL.Image.fromarray(img.astype('uint8'))
    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    img = trans(img)
    return img


# 2 classes based on training set
classes = ['circle', 'retrangle']
o1 = net(Variable(array2tensor(circle).unsqueeze(0)))
o2 = net(Variable(array2tensor(retrangle).unsqueeze(0)))
_, idx1 = torch.max(o1.data, 1)
_, idx2 = torch.max(o2.data, 1)
print(classes[idx1[0]], classes[idx2[0]])
