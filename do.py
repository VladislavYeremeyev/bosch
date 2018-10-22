from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.linalg as nl
from scipy import misc
from PIL import Image

import matplotlib.pyplot as plt

from detection import detectStructures

bp = "/home/kerofeev/work/tests/Python/bosch/data/cur/"
i = Image.open("/home/kerofeev/neural-style/examples/inputs/starry_night.jpg")
i = Image.open("/home/kerofeev/neural-style/out.png")

d = detectStructures(bf+"ZB0.jpg")


