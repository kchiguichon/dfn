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

The dataset used for this project is the Quizbowl Data set from 2015 provided by UMASS (no affiliation) and can be accesed via https://people.cs.umass.edu/~miyyer/data/question_data.tar.gz

The general task for this dataset is that of factoid question answering. This is modeled in this project as a multi-class classification problem since the set of all answers is fixed.

More details can be found at https://sites.google.com/view/qanta/home

There is an automated script for downloading the dataset found in this prohect in src/util/download_data.sh

To run it simply enter
```
bash ./src/util/download_data.sh
```
into a terminal with bash installed.

# Code Overview

The code is divided into a few different packages. Their contents and uses are listed below.

#### ./src 

This directory contains all of the code for this project. However the python files contained as a direct child of this directory are utilized to train, and evaluate different the models tested in this project.

#### ./src/util

This directory contains utilities for creating the necessary data files for training and evaluating as well as utilities for creating training analysis plots. 

In order to run train.py and eval.py from the ./src/ folder it is necessary to run the following utilities to create the relevant train/dev/test data partitions as well as label dictionary.

```
python ./src/util/create_data_partitions.py --questions-path ./data/questions.csv --save-dir ./data/
python ./src/util/create_label_dict.py --questions-path ./data/questions.csv --output-path ./data/answers.json
```


## Train and Evaluate

Both ./src/train.py and ./src/eval.py print out their usage menu with the -h flag. This is useful to see the parameters both scripts take.

#### Train a model

An example command to start training the model.
```
python ./src/train.py --train ./data/train.csv --dev ./data/dev.csv --labels ./data/answers.json --model-type dan --embeddings ./data/glove.6B.50d.txt --embed-dim 50 --num-layers 4 --checkpoint-path ./serialization_dirs/dan --num-epochs 20
```

#### Evaluating model

An example command to evaluate a trained model.

```
python ./src/eval.py --model ./serialization_dirs/dan --test ./data/test.csv --labels ./data/answers.json
```

## Extra Scripts

The ./src/util/plot_statistics.py script takes in the serialization_dir for an arbitrary amount of models and plots their training statistics for analysis.

An example run is
```
python ./src/util/plot_statistics.py ./serialization_dirs/dan/
```

But as stated you can feed any amount of serializatoin directories to this script and it will plot comparison curves for all directories given to it.

./src/util/constants.py contains global constants used throughout this project for easy modificaition.

# Author
All code written for this project was written by it's sole author: Kenneth Chiguichon. This code is licensed with Apache License 2.0, included in the LICENSE file.

For any inquiries on this project email the author at kchiguichon@cs.stonybrook.edu.
