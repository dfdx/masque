"""
Utilities for visual debugging of learning.
Partially based on:
  http://yosinski.com/media/papers/Yosinski2012VisuallyDebuggingRestrictedBoltzmannMachine.pdf
"""

import numpy as np
import matplotlib.pylab as plt
from PIL import Image

def sigmoid(xx):
    return .5 * (1 + np.tanh(xx / 2.))

    