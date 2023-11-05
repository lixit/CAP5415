from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import argparse


def knn(X_train, y_train, X_test, y_test, k):
    """
    k-nearest neighbors algorithm
    """
    correct = 0
    # loop all test data
    for i in range(X_test.shape[0]):
        test = X_test[i]
        distance_with_index = []
        # compute distance with all training set
        for j in range(y_train.shape[0]):
            train = X_train[j]
            l2_norm = np.sqrt(np.square(test - train).sum())
            distance_with_index.append((l2_norm, j))
        
        # sort based on the distance
        distance_with_index.sort(key=lambda x: x[0])

        # get the first k
        k_nearest = distance_with_index[:k]
        # add all labels to the list, include duplicates
        labels = []
        for dist, idx in k_nearest:
            labels.append(y_train[idx])

        # find the most frequent label
        label, count = np.unique(labels, return_counts=True)
        # argmax returns the index of the maximum value
        predict = label[np.argmax(count)]
        ground_truth = y_test[i]
        if predict == ground_truth:
            correct += 1

    print("Accuracy: ", correct / X_test.shape[0])
        

def main(FLAGS):
    digits = load_digits()

    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=500, random_state=42)

    print("train size: ", X_train.shape[0])
    print("test size: ", X_test.shape[0])
    print("k: ", FLAGS.k)

    knn(X_train, y_train, X_test, y_test, FLAGS.k)

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--k',
                        type=int, default=1,
                        help='K value')
    
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)