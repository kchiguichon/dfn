#!/bin/bash

set -e
set -x

# Dataset provided by UMASS (no affiliation)
# URL to original work https://people.cs.umass.edu/~miyyer/qblearn/index.html
DATA_URL=https://people.cs.umass.edu/~miyyer/data/question_data.tar.gz

cd data/
wget $DATA_URL
tar xvzf $(basename $DATA_URL)
rm $(basename $DATA_URL)
