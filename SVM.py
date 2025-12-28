#This function will train a linear SVM for every category (i.e. one vs all)
#and then use the learned linear classifiers to predict the category of
#every test image. Every test feature will be evaluated with all SVMs
#and the most confident SVM will "win". Confidence, or distance from the
#margin, is W*X + B where '*' is the inner product or dot product and W and
#B are the learned hyperplane parameters.

from sklearn.svm import LinearSVC
import numpy as np

def svm_classify(train_image_feats, train_labels, test_image_feats=None):
    classes = list(set(train_labels))
    svms = {}

    for c in classes:
        y = np.array([1 if lbl == c else 0 for lbl in train_labels])
        clf = LinearSVC()
        clf.fit(train_image_feats, y)
        svms[c] = clf

    predicted = None
    if test_image_feats is not None:
        print("Testing SVMs on test images")
        predicted = []
        for feat in test_image_feats:
            scores = {c: clf.decision_function([feat])[0] for c, clf in svms.items()}
            best_class = max(scores, key=scores.get)
            predicted.append(best_class)
        print("Done testing")

    return svms, predicted

# image_feats is an N x d matrix, where d is the dimensionality of the
#  feature representation.
# train_labels is an N x 1 cell array, where each entry is a string
#  indicating the ground truth category for each training image.
# test_image_feats is an M x d matrix, where d is the dimensionality of the
#  feature representation. You can assume M = N unless you've modified the
#  starter code.
# predicted_categories is an M x 1 cell array, where each entry is a string
#  indicating the predicted category for each test image.
