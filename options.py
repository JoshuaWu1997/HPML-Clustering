import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()

# -------------------------------------------------------------------------
# Data input settings
parser.add_argument("--dataset", type=str, default="CIFAR", help="path to dataset")
# -------------------------------------------------------------------------
# Model params
parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
# -------------------------------------------------------------------------
# Optimization / training params
parser.add_argument("--alg", type=str, default="gmm", help="")
parser.add_argument("--kernel", type=str, default="l2", help="")
parser.add_argument('--kernel_cuda', default=False, type=str2bool, help='Use my_cuda')
parser.add_argument("--device", type=str, default="cuda", help="")
parser.add_argument("--gpu_type", type=str, default="local", help="")
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Load Models
params, unparsed = parser.parse_known_args()
