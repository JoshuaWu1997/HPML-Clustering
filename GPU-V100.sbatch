#!/bin/bash
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --cpus-per-task=8                # uses 1 compute core per task
#SBATCH --gres=gpu:v100:1  ## To request specific GPU (v100 or V1008000)
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=V100
#SBATCH --output=V100.out

python test.py --dataset=CIFAR --alg="kmeans++" --kernel=l2 --kernel_cuda=False --device=cuda --gpu_type=V100
python test.py --dataset=CIFAR --alg="kmeans++" --kernel=l2 --kernel_cuda=True --device=cuda --gpu_type=V100
python test.py --dataset=CIFAR --alg="kmeans++" --kernel=l2 --kernel_cuda=False --device=cpu --gpu_type=V100

python test.py --dataset=CIFAR --alg="kmeans++" --kernel=m3 --kernel_cuda=False --device=cuda --gpu_type=V100
python test.py --dataset=CIFAR --alg="kmeans++" --kernel=m3 --kernel_cuda=True --device=cuda --gpu_type=V100
python test.py --dataset=CIFAR --alg="kmeans++" --kernel=m3 --kernel_cuda=False --device=cpu --gpu_type=V100

python test.py --dataset=CIFAR --alg="kmeans++" --kernel=m4 --kernel_cuda=False --device=cuda --gpu_type=V100
python test.py --dataset=CIFAR --alg="kmeans++" --kernel=m4 --kernel_cuda=True --device=cuda --gpu_type=V100
python test.py --dataset=CIFAR --alg="kmeans++" --kernel=m4 --kernel_cuda=False --device=cpu --gpu_type=V100


python test.py --dataset=Synthetic --alg="kmeans++" --kernel=l2 --kernel_cuda=False --device=cuda --gpu_type=V100
python test.py --dataset=Synthetic --alg="kmeans++" --kernel=l2 --kernel_cuda=True --device=cuda --gpu_type=V100
python test.py --dataset=Synthetic --alg="kmeans++" --kernel=l2 --kernel_cuda=False --device=cpu --gpu_type=V100

python test.py --dataset=Synthetic --alg="kmeans++" --kernel=m3 --kernel_cuda=False --device=cuda --gpu_type=V100
python test.py --dataset=Synthetic --alg="kmeans++" --kernel=m3 --kernel_cuda=True --device=cuda --gpu_type=V100
python test.py --dataset=Synthetic --alg="kmeans++" --kernel=m3 --kernel_cuda=False --device=cpu --gpu_type=V100

python test.py --dataset=Synthetic --alg="kmeans++" --kernel=m4 --kernel_cuda=False --device=cuda --gpu_type=V100
python test.py --dataset=Synthetic --alg="kmeans++" --kernel=m4 --kernel_cuda=True --device=cuda --gpu_type=V100
python test.py --dataset=Synthetic --alg="kmeans++" --kernel=m4 --kernel_cuda=False --device=cpu --gpu_type=V100


python test.py --dataset=Synthetic --alg="GMM" --kernel=l2 --kernel_cuda=False --device=cuda --gpu_type=V100
python test.py --dataset=Synthetic --alg="GMM" --kernel=l2 --kernel_cuda=False --device=cpu --gpu_type=V100

python test.py --dataset=Synthetic --alg="GMM" --kernel=m3 --kernel_cuda=False --device=cuda --gpu_type=V100
python test.py --dataset=Synthetic --alg="GMM" --kernel=m3 --kernel_cuda=False --device=cpu --gpu_type=V100

python test.py --dataset=Synthetic --alg="GMM" --kernel=m4 --kernel_cuda=False --device=cuda --gpu_type=V100
python test.py --dataset=Synthetic --alg="GMM" --kernel=m4 --kernel_cuda=False --device=cpu --gpu_type=V100