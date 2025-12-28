# Name : Aurangzeb
# Roll Number : BSAI23021 
# Assignment : 3



#This function will predict the category for every test image by finding
#the training image with most similar features. Instead of 1 nearest
#neighbor, you can vote based on k nearest neighbors which will increase
#performance (although you need to pick a reasonable value for k).

import numpy as np
from collections import Counter

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k=3):
    predicted_categories = []
    
    for test_feat in test_image_feats:
        # compute L2 distances to all training features
        dists = np.linalg.norm(train_image_feats - test_feat, axis=1)
        # get k nearest indices
        nearest_idx = np.argsort(dists)[:k]
        # get their labels
        nearest_labels = [train_labels[i] for i in nearest_idx]
        # majority vote
        most_common = Counter(nearest_labels).most_common(1)[0][0]
        predicted_categories.append(most_common)
    
    return predicted_categories

# image_feats is an N x d matrix, where d is the dimensionality of the
#  feature representation.
# train_labels is an N x 1 cell array, where each entry is a string
#  indicating the ground truth category for each training image.
# test_image_feats is an M x d matrix, where d is the dimensionality of the
#  feature representation. You can assume M = N unless you've modified the
#  starter code.
# predicted_categories is an M x 1 cell array, where each entry is a string
#  indicating the predicted category for each test image.