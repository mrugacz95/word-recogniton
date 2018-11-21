import re
from argparse import ArgumentParser

import glob2
import librosa
import numpy as np
import progressbar
import soundfile as sf
from dtw import dtw
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def extract_number(sample_name):
    return re.findall('\d', sample_name)[-1]


def normalize(data):
    return data / np.linalg.norm(data)


def compare_plots():
    data, samplerate = sf.read('numbers/train/AK1C1_0_.WAV')
    plt.plot(normalize(data), label='train', alpha=0.8)
    data, samplerate = sf.read('numbers/test/sample_0.wav')
    plt.plot(normalize(data), label='test', alpha=0.8)
    plt.legend()
    plt.show()
    return


number_of_classes = 0
class_to_number = {}


def collect_data(dataset_name):
    global class_to_number
    global number_of_classes
    X = []
    y = []
    for path in glob2.glob(f'{dataset_name}/**/*.WAV'):
        sample_class = path.split('/')[1]
        data, sample_rate = sf.read(path)
        data = normalize(data)
        X.append(np.array(librosa.feature.mfcc(data, sample_rate, n_mfcc=13)).T)
        if sample_class not in class_to_number:
            class_to_number[sample_class] = number_of_classes
            number_of_classes += 1
        class_number = class_to_number[sample_class]
        y.append(class_number)
    return train_test_split(X, y, test_size=0.25)


def main(dataset):
    X_train, X_test, y_train, y_test = collect_data(dataset)

    y_true = []
    y_pred = []
    bar = progressbar.ProgressBar(max_value=len(y_test))
    for sample, sample_class in bar(zip(X_test, y_test)):
        min_dist = np.inf
        predicted_class = None
        for other, other_class in zip(X_train, y_train):  # kNN with train train
            distance, _, _, _ = dtw(sample, other, dist=lambda x, y: np.linalg.norm(x.T - y.T, ord=1))
            if distance < min_dist:
                min_dist = distance
                predicted_class = other_class
        y_true.append(sample_class)
        y_pred.append(predicted_class)
    mat = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(1, 1)
    ax.matshow(mat, cmap='Blues')

    for (i, j), z in np.ndenumerate(mat):
        ax.text(j, i, str(z), ha='center', va='center', size='x-small')
    ticks = []
    ticks_pos = []
    for class_name in class_to_number.keys():
        ticks_pos.append(class_to_number[class_name])
        ticks.append(class_name)
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(ticks)
    ax.set_yticks(ticks_pos)
    ax.set_yticklabels(ticks)
    print(y_pred)
    print(y_true)
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='Name of dataset', default='commands', required=False)
    args = parser.parse_args()
    main(args.dataset)
