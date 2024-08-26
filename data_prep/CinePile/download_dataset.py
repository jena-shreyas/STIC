import os
from os.path import dirname as osd
import random
import subprocess
random.seed(42)

import pandas as pd
import numpy as np

from datasets import load_dataset

## load the dataset
DATA_ROOT=osd(osd(os.getcwd()))+"datasets/CinePile"
cinepile = load_dataset("tomg-group-umd/cinepile", cache_dir=DATA_ROOT)