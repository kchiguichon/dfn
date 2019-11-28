This file is best viewed in Markdown reader (eg. https://jbt.github.io/markdown-editor/)

# Overview

TODO: An overview of the project.


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
python ./src/train.py --train ./data/train.csv --dev ./data/dev.csv --labels ./data/answers.json --model dan --embeddings ./data/glove.6B/glove.6B.50d.txt --embed-dim 50 --num-layers 4
```

#### Predict with model

TODO: Explain how to predict with model.


## Extra Scripts

Explain Utils and other misc scripts.

# General results

TODO: Overview of results.
