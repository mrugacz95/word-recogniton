import re

import glob2
import librosa
import soundfile as sf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from dtw import dtw


def extract_number(sample_name):
    return re.findall('\d', sample_name)[-1]


def normalize(data):
    return data  / np.linalg.norm(data)


def compare_plots():
    data, samplerate = sf.read('train/AK1C1_0_.WAV')
    plt.plot(normalize(data), label='train', alpha=0.8)
    data, samplerate = sf.read('test/sample_0.wav')
    plt.plot(normalize(data), label='test', alpha=0.8)
    plt.legend()
    plt.show()
    return


def main():
    X = []
    y = []
    for path in glob2.glob('train/*'):
        data, samplerate = sf.read(path)
        data = normalize(data)
        X.append(np.array(librosa.feature.mfcc(data, samplerate, n_mfcc=13)).T)
        y.append(extract_number(path))

    # split train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0)
    # or load train
    # X_train = X
    # y_train = y
    # X_test = []
    # y_test = []
    for path in glob2.glob('test/*'):
        data, samplerate = sf.read(path)
        data = normalize(data)
        X_test.append(np.array(librosa.feature.mfcc(data, samplerate, n_mfcc=13)).T)
        y_test.append(extract_number(path))

    y_true = []
    y_pred = []
    for sample, sample_class in zip(X_test, y_test):
        min_dist = np.inf
        predicted_class = None
        for other, other_class in zip(X_train, y_train):  # kNN with train train
            distance, _, _, _ = dtw(sample, other, dist=lambda x, y: np.linalg.norm(x.T - y.T, ord=1))
            if distance < min_dist:
                min_dist = distance
                predicted_class = other_class
            print(sample_class, other_class, distance)
        y_true.append(sample_class)
        y_pred.append(predicted_class)
    mat = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(1, 1)
    ax.matshow(mat, cmap='Blues')

    for (i, j), z in np.ndenumerate(mat):
        ax.text(j, i, str(z), ha='center', va='center', size='x-small')

    print(y_pred)
    print(y_true)
    plt.show()


if __name__ == '__main__':
    main()
