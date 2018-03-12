import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def imshow(img):
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    ax.axis("off")
    plt.imshow(img / 255)
    plt.show()


class Shape:
    def __init__(self):
        self.colors = [
            (0, 0, 255),  # r
            (0, 255, 0),  # g
            (255, 0, 0),  # b
            (0, 156, 255),  # o
            (128, 128, 128),  # gray
            (0, 255, 255)  # cyan
        ]
        self.canvas_size = 100

    def make(self, model):
        img = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.float32) * 255
        color = self.colors[random.randint(0, 5)]
        center = [random.randint(40, 60), random.randint(40, 60)]
        object_size = random.randint(10, 40)
        if model == 'retrangle':
            start = (center[0] - object_size, center[1] - object_size)  # painting start point
            end = (center[0] + object_size, center[1] + object_size)
            cv2.rectangle(img, start, end, color, -1)
        if model == 'circle':
            cv2.circle(img, (center[0], center[1]), object_size, color, -1)
        return img


# creator = Shape()
# circle = creator.make('circle')
# imshow(circle)
# retrangle = creator.make('retrangle')
# imshow(retrangle)
