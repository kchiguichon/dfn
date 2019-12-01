This file is best viewed in Markdown reader (eg. https://jbt.github.io/markdown-editor/)

# Overview

This repo contains the implementation for a new type of deep compositional network. Inspiration and motivation was from the work done by Iyyer et al. in their paper "Deep Unordered Composition Rivals Syntactic Methods for Text Classification" (https://www.aclweb.org/anthology/P15-1162/). 

This work provides a new aggregation layer which aims at capturing sequential information. Using a different representation for word vectors and subsequently also computing a different representation for sequences.

While due to computing fourier transforms on input data the model is slower than a typical DAN implementation it is still much faster to train that heavier RNN based models while also converging faster than DAN and outperforming it in accuracy.


# Installation

This project is implemented in python 3.6 and tensorflow 2.0. Follow these steps to setup your environment:

1. [Download and install Conda](http://https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")
2. Create a Conda environment with Python 3.6

```
conda create -n nlp-project python=3.6
```

3. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code.
```
conda activate nlp-project
```
4. Install the requirements:
```
pip install -r requirements.txt
```

5. Download glove wordvectors:
```
./src/util/download_glove.sh
```


# Data

TODO: Explain data.


# Code Overview


TODO: Explain code structure and design and give overview of how everything fits together.


## Train and Predict

TODO: Explain train and predict scripts.

#### Train a model

An example command to start training the model.
```
python ./src/train.py --train ./data/train.csv --dev ./data/dev.csv --labels ./data/answers.json --model-type dan --embeddings ./data/glove.6B.50d.txt --embed-dim 50 --num-layers 4 --checkpoint-path ./serialization_dirs/dan --num-epochs 20
```

#### Predict with model

TODO: Explain how to predict with model.


## Extra Scripts

Explain Utils and other misc scripts.

# General results

TODO: Overview of results.
