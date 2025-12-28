# Name : Aurangzeb
# Roll Number : BSAI23021
# Assignment : 3

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import pandas as pd
import os
import glob

# this part finds all the images and gives their labels
def get_image_paths(data_path, categories):
    train_image_paths, train_labels = [], []
    for cat in categories:
        imgs = glob.glob(data_path+'train/'+cat+'/*.*')
        train_image_paths = train_image_paths + imgs
        train_labels = train_labels + [cat]*len(imgs)

    test_image_paths, test_labels = [], []
    for cat in os.listdir(data_path+'test/'):
        imgs = glob.glob(data_path+'test/'+cat+'/*.*')
        test_image_paths = test_image_paths + imgs
        test_labels = test_labels + [cat]*len(imgs)

    return np.array(train_image_paths), np.array(test_image_paths), np.array(train_labels), np.array(test_labels)  

# this part makes the confusion matrix to see where model is right or wrong
def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    labels = unique_labels(y_true, y_pred)

    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    if classes is None:
        display_labels = labels
    else:
        try:
            display_labels = [c for c in classes if c in labels]
            if len(display_labels) == 0:
                display_labels = labels
            else:
                display_labels = np.array(display_labels)
        except Exception:
            display_labels = labels

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=display_labels, yticklabels=display_labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    try:
        thresh = cm.max() / 2.
    except Exception:
        thresh = 0.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center",
                    color="white" if val > thresh else "black")
    fig.tight_layout()
    return ax

# this one counts true positives, false positives, true negatives, and false negatives
def perf_measure(y_actual, y_hat):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        elif y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        elif y_actual[i]==y_hat[i]==0:
            TN += 1
        elif y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return [TP, FP, TN, FN]

# this part shows all final scores like accuracy, f1 score and confusion matrix
def display_results(test_labels, categories, predicted_categories, output_dir='results', file_prefix='results'):
    labels_are_numeric = np.issubdtype(test_labels.dtype, np.integer)
    cols = ['Category', 'TP', 'FP', 'TN', 'FN']
    df = pd.DataFrame(columns=cols)

    # here it checks each category one by one and makes the table
    for i, el in enumerate(categories):
        if labels_are_numeric:
            temp_y_test = (test_labels == i).astype(int)
            temp_preds = (predicted_categories == i).astype(int)
        else:
            temp_y_test = (test_labels == el).astype(int)
            temp_preds = (predicted_categories == el).astype(int)

        row = [el] + perf_measure(temp_y_test, temp_preds)
        df = pd.concat([df, pd.DataFrame([row], columns=cols)], ignore_index=True)

    print(df, '\n\n')

    # here it changes text labels into numbers so it can plot better
    if not labels_are_numeric:
        for i in range(len(categories)):
            test_labels[test_labels == categories[i]] = i
            predicted_categories[predicted_categories == categories[i]] = i

    test_labels = test_labels.astype(int)
    predicted_categories = predicted_categories.astype(int)

    class_names = np.array(categories)
    ax = plot_confusion_matrix(test_labels, predicted_categories, classes=class_names)

    os.makedirs(output_dir, exist_ok=True)
    fig = ax.figure
    fig.savefig(os.path.join(output_dir, f'{file_prefix}_confusion_matrix.png'))
    fig.show()

    accuracy = np.mean(test_labels == predicted_categories)
    f1 = f1_score(y_true=test_labels, y_pred=predicted_categories, average='macro')

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'F1 Score (macro): {f1:.4f}')

    return accuracy, f1
