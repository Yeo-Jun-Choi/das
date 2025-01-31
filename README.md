# Are GCNs Really Necessary for Recommendation?: How Degree Beats GCN

## Requirements
python 3.8.18, cuda 11.8, and the following installations:
```
# installation
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
pip install torch-geometric
pip install six
pip install pandas

```

## Training
You can train the models Amazon-books, Gowalla, Movielens-1M, and Yelp18 by entering the following command line.

```
cd bash

# Amazon-books
sh run_amazon.sh

# Gowalla
sh run_gowalla.sh

# Movielens-1m
sh run_ml1m.sh

# Yelp18
sh run_yelp.sh
```