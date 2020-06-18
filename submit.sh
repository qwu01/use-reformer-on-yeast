#!/bin/bash
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out

module load python/3.7

ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate

##python ../downloads/product_key_memory-0.1.10/setup.py install
##python ../downloads/axial_positional_embedding-0.2.1/setup.py install
##python ../downloads/reformer_pytorch-1.0.2/setup.py install
pip install --no-index torch torchvision numpy scipy matplotlib pandas nltk scikit-learn tqdm
pip install --no-index ../downloads/transformers-2.11.0-py3-none-any.whl
pip install --no-index ../downloads/torchtext-0.6.0-py3-none-any.whl

python tmp.py

deactivate

rm -r $ENVDIR
