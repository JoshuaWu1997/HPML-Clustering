import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()

# -------------------------------------------------------------------------
# Data input settings
parser.add_argument("--dataset", type=str, default="Synthetic", help="path to dataset")
# -------------------------------------------------------------------------
# Model params
parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
# -------------------------------------------------------------------------
# Optimization / training params
parser.add_argument('--train_batch_size', default=128, type=int, help='Batch size (Adjust base on GPU memory)')
parser.add_argument("--alg", type=str, default="kmeans++", help="")
parser.add_argument("--kernel", type=str, default="m3", help="")
parser.add_argument('--kernel_cuda', default=True, type=str2bool, help='Use my_cuda')
parser.add_argument("--device", type=str, default="cuda", help="")
parser.add_argument("--gpu_type", type=str, default="V100", help="")
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Load Models
params, unparsed = parser.parse_known_args()
