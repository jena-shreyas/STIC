import os
from os.path import dirname as osd, join as osj
import random
import subprocess
random.seed(42)

import pandas as pd
import numpy as np

from datasets import load_dataset

## load the dataset
DATA_ROOT=osj(osd(osd(os.getcwd())),"datasets/CinePile")
print(DATA_ROOT)

os.makedirs(DATA_ROOT, exist_ok=True)
cinepile = load_dataset("tomg-group-umd/cinepile", cache_dir=DATA_ROOT)