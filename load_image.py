import argparse
import json
import os

import torch
from cyy_naive_pytorch_lib.dataset import DatasetUtil
from cyy_naive_pytorch_lib.dataset_collection import DatasetCollection
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
args = parser.parse_args()
