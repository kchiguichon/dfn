import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_plot_data(filepath):
    return json.load(open(os.path.join(filepath, f'plot_data.json'), 'r', encoding='utf8'))

def make_plots(plot_paths):
    plot_data_list = [load_plot_data(arg) for arg in plot_paths]
    fig, axs = plt.subplots(2, 2)
    plot_categories = ['training_loss', 'training_accuracy', 'validation_loss', 'validation_accuracy']
    for i in range(len(plot_categories)):
        axis = axs[i//2, i % 2]
        axis.set_title(plot_categories[i])
        for j in range(len(plot_data_list)):
            axis.plot(range(len(plot_data_list[j][plot_categories[i]])), plot_data_list[j][plot_categories[i]], 'C{}-'.format(j), label=os.path.dirname(plot_paths[j]).split(r'/')[-1])
        axis.legend(loc='best')
    plt.subplots_adjust(hspace=0.4)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Script to make analysis plots for comparing models.""")
    parser.add_argument('model_checkpoint_paths', nargs='*', help='Model checkpoint paths to include in graphs.')
    args = parser.parse_args()

    make_plots(args.model_checkpoint_paths)