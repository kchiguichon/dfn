import pandas as pd
import argparse
from pathlib import Path

def read_data(filepath, category = None):
    df = pd.read_csv(open(filepath, 'r'), quotechar='"')
    return df if category is None else df.loc[df['Category'] == category]

def partition_data(df):
    partitions = {'train': None, 'dev' : None, 'test' : None}
    for partition in partitions.keys():
        partitions[partition] = df.loc[df['Fold'] == partition]
    return partitions

def save_partitions(partitions, directory):
    dir_path = Path(directory)
    if not dir_path.exists():
        dir_path.mkdir()
    elif dir_path.is_file():
        raise ValueError('Save directory path provided is not a directory: {}'.format(directory))
    for name, df in partitions.items():
        df.to_csv(directory + name + '.csv', index=False, quotechar='"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        A simple program to partition the data received from QANTA dataset in csv format. 
    """)

    parser.add_argument('--questions-path', help='Path to questions.csv file.')
    parser.add_argument('--category', help='Category of questions. If not provided all categories will be considered.')
    parser.add_argument('--save-dir', help='Save directory to place partitions in.')
    args = parser.parse_args()

    data = read_data(args.questions_path, category=args.category)
    partitions = partition_data(data)
    save_directory = args.save_dir.strip()
    save_directory =  save_directory if save_directory[-1] in {'/', '\\'} else save_directory + '\\'
    save_partitions(partitions, save_directory)
    