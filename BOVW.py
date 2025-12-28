# Name : Aurangzeb
# Roll Number : BSAI23021 
# Assignment : 3

import cv2
import numpy as np
import pickle
from PIL import Image
from sklearn.cluster import KMeans
import random
import os
import matplotlib.pyplot as plt

# this part takes small image parts and groups them to make a visual dictionary
def build_vocabulary(image_paths, vocab_size, num_descriptors=200):
    all_features = []
    for path in image_paths:
        img = Image.open(path).convert("L")
        img = np.array(img)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            if len(des) > num_descriptors:
                des = np.array(des[random.sample(range(len(des)), num_descriptors)])
            all_features.append(des)
    all_features = np.vstack(all_features)
    kmeans = KMeans(n_clusters=vocab_size, random_state=42)
    kmeans.fit(all_features)
    vocab = kmeans.cluster_centers_
    return vocab

# this part makes bag of words features for every image using vocab
def get_bags_of_sifts(image_paths, vocab_path='vocab.pkl', num_descriptors=400, labels=None, visualize=False, vis_dir="results/visualizations"):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = vocab.shape[0]
    image_feats = []
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
        visualized_classes = set()
    for path in image_paths:
        img = Image.open(path).convert("L")
        img = np.array(img)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        if des is None:
            des = np.zeros((1, 128))
        if len(des) > num_descriptors:
            des = np.array(des[random.sample(range(len(des)), num_descriptors)])
        dists = np.linalg.norm(des[:, np.newaxis] - vocab[np.newaxis, :], axis=2)
        words = np.argmin(dists, axis=1)
        hist = np.zeros(vocab_size)
        for w in words:
            hist[w] += 1
        hist = hist / np.linalg.norm(hist) if np.linalg.norm(hist) != 0 else hist
        image_feats.append(hist)
        if visualize:
            class_name = os.path.basename(os.path.dirname(path))
            if class_name not in visualized_classes:
                os.makedirs(os.path.join(vis_dir, class_name), exist_ok=True)
                keypoint_img = cv2.drawKeypoints(img, random.sample(kp, min(50, len(kp))) if kp else [], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imwrite(os.path.join(vis_dir, class_name, os.path.basename(path) + "_keypoints.png"), keypoint_img)
                plt.figure()
                plt.bar(range(len(hist)), hist)
                plt.savefig(os.path.join(vis_dir, class_name, os.path.basename(path) + "_histogram.png"))
                plt.close()
                visualized_classes.add(class_name)
    image_feats = np.array(image_feats)
    return image_feats

# this part draws a histogram that shows word counts for an image
def plot_histogram(hist, save_path):
    plt.figure()
    plt.bar(range(len(hist)), hist)
    plt.savefig(save_path)
    plt.close()

# this part shows sift keypoints drawn on the image
def visualize_sift_keypoints(image_path, save_path, num_keypoints=50):
    img = Image.open(image_path).convert("L")
    img = np.array(img)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    if kp is not None:
        if len(kp) > num_keypoints:
            kp = random.sample(kp, num_keypoints)
    img_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(save_path, img_kp)
