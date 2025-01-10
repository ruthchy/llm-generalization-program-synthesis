import numpy as np

# Implementation of ReGAL LOGO primitives using Matplotlib
class ReGALLOGOPrimitives:
    '''Defining all the LOGO ReGAL primitives'''

    def __init__(self):
        self.x, self.y = 0, 0  # Current position
        self.angle = 0         # Current angle in degrees
        self.is_drawing = True
        self.path = []         # List of drawn lines
        self.pen_up_path = []  # List of pen-up moves

    def _add_to_path(self, x1, y1, x2, y2):
        if self.is_drawing:
            self.path.append(((x1, y1), (x2, y2)))
        else:
            self.pen_up_path.append(((x1, y1), (x2, y2)))

    def forward(self, distance):
        x2 = self.x + distance * np.cos(np.radians(self.angle))
        y2 = self.y + distance * np.sin(np.radians(self.angle))
        self._add_to_path(self.x, self.y, x2, y2)
        self.x, self.y = x2, y2

    def left(self, angle):
        self.angle = (self.angle + angle) % 360

    def right(self, angle):
        self.angle = (self.angle - angle) % 360

    def penup(self):
        self.is_drawing = False

    def pendown(self):
        self.is_drawing = True

    def teleport(self, x, y):
        self.penup()
        self.x, self.y = x, y
        self.pendown()

    def heading(self, angle):
        self.angle = angle % 360

    def isdown(self):
        return self.is_drawing

