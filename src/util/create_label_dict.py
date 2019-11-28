import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def build_label_dict(questions_path):
    questions = set(pd.read_csv(open(questions_path, 'r'), quotechar='"')['Answer'].apply(lambda x: np.array(re.sub(r'\s+', '_', x))))
    return dict(zip(questions, list(range(len(questions)))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple program to create label dictionary')
    parser.add_argument('--questions-path', help='Path to question dataset.', default='.\\data\\questions.csv')
    parser.add_argument('--output-path', help='Path to save labels dictionary to', default='.\\data\\answers.json')
    args = parser.parse_args()
    question_to_id = build_label_dict(args.questions_path)
    json.dump(question_to_id, open(args.output_path, 'w', encoding='utf8'))
