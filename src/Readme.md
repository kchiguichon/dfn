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

5. Download spacy model
```
python -m spacy download en_core_web_sm
```

6. Download glove wordvectors:
```
./download_glove.sh
```


# Data

TODO: Explain data.


# Code Overview


TODO: Explain code structure and design and give overview of how everything fits together.


## Train and Predict

TODO: Explain train and predict scripts.

#### Train a model

TODO: Explain how to train model.

#### Predict with model

TODO: Explain how to predict with model.


## Extra Scripts

Explain Utils and other misc scripts.

# General results

TODO: Overview of results.
