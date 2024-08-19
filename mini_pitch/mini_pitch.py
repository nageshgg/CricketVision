import cv2
import sys
sys.path.append('../')
import sre_constants

class MiniPitch():
    def __init__(self, frame):
        self.drawing_ractangle_wdth = 250
        self.drawing_ractangle_height = 400
        self.bufffer = 30
        self.padding = 10

    def set_canvas_