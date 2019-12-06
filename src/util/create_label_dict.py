# Copyright 2019 Kenneth Chiguichon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def build_label_dict(questions_path, category = None):
    questions = pd.read_csv(open(questions_path, 'r'), quotechar='"')
    if category is not None:
        questions = questions.loc[questions['Category'] == category]
    questions = set(questions['Answer'].apply(lambda x: np.array(re.sub(r'\s+', '_', x))))
    return dict(zip(questions, list(range(len(questions)))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        A simple program to create label dictionary used for converting label to ID and vice versa.
    """)
    parser.add_argument('--questions-path', help='Path to question dataset.', default='.\\data\\questions.csv')
    parser.add_argument('--category', help='Category of questions. If not provided all categories will be considered.')
    parser.add_argument('--output-path', help='Path to save labels dictionary to', default='.\\data\\answers.json')
    args = parser.parse_args()
    question_to_id = build_label_dict(args.questions_path, category=args.category)
    json.dump(question_to_id, open(args.output_path, 'w', encoding='utf8'))
