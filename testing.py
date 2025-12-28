import os
import pickle
import numpy as np
from BOVW import get_bags_of_sifts
from KNN import nearest_neighbor_classify
from utils import display_results
from sklearn.metrics import accuracy_score, f1_score

def _load_knn_train_features_from_path(image_feats_path):
    if image_feats_path is None:
        raise ValueError("image_feats_path is required for BoVW KNN training features.")
    if not os.path.isfile(image_feats_path):
        raise FileNotFoundError(f"KNN train file not found: {image_feats_path}")
    with open(image_feats_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "features" in data and "labels" in data:
        feats = np.array(data["features"])
        labels = np.array(data["labels"])
    else:
        raise ValueError(f"Unexpected format in {image_feats_path}. Expected dict with keys 'features' and 'labels'.")
    return feats, labels

def testing_onOneImage(path_to_an_image, type_, path_to_model=None, path_to_BOW_VOCAB=None, path_to_image_feats=None):
    if not os.path.isfile(path_to_an_image):
        print("Image not found:", path_to_an_image)
        return None
    if path_to_BOW_VOCAB is None:
        raise ValueError("This testing script only supports Bag-of-SIFT (provide path_to_BOW_VOCAB).")
    if not os.path.isfile(path_to_BOW_VOCAB):
        raise FileNotFoundError("BoVW vocab not found: " + path_to_BOW_VOCAB)

    clf_type = type_.strip().lower()
    true_label = os.path.basename(os.path.dirname(path_to_an_image))
    test_feats = get_bags_of_sifts([path_to_an_image], vocab_path=path_to_BOW_VOCAB, labels=None)
    feat = test_feats[0]

    if clf_type == "knn":
        train_feats, train_labels = _load_knn_train_features_from_path(path_to_image_feats)
        pred = nearest_neighbor_classify(train_feats, train_labels, feat.reshape(1, -1), k=3)[0]
    elif clf_type == "svm":
        if path_to_model is None or not os.path.isfile(path_to_model):
            raise FileNotFoundError("SVM model file not found: " + str(path_to_model))
        with open(path_to_model, "rb") as f:
            svms = pickle.load(f)
        scores = {c: clf.decision_function([feat])[0] for c, clf in svms.items()}
        pred = max(scores, key=scores.get)
    else:
        raise ValueError("Unknown type. Use 'KNN' or 'SVM'.")

    correct = (pred == true_label)
    print("Image:", path_to_an_image)
    print("True label   :", true_label)
    print("Predicted    :", pred)
    print("Correct?     :", "YES" if correct else "NO")
    print("-" * 40)
    return pred, true_label, correct

def testing_AllImages(path_to_image_folder, type_, path_to_model=None, path_to_BOW_VOCAB=None, path_to_image_feats=None):
    if path_to_BOW_VOCAB is None:
        raise ValueError("This testing script only supports Bag-of-SIFT (provide path_to_BOW_VOCAB).")
    if not os.path.isdir(path_to_image_folder):
        raise FileNotFoundError("Image folder not found: " + path_to_image_folder)

    # Gather all test image paths and labels
    image_paths = []
    true_labels = []
    for cls in sorted(os.listdir(path_to_image_folder)):
        cls_dir = os.path.join(path_to_image_folder, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_paths.append(os.path.join(cls_dir, fname))
                true_labels.append(cls)

    if len(image_paths) == 0:
        print("No images found in:", path_to_image_folder)
        return None

    print(f"Computing BoVW features for {len(image_paths)} test images...")
    test_feats = get_bags_of_sifts(image_paths, vocab_path=path_to_BOW_VOCAB, labels=true_labels)

    clf_type = type_.strip().lower()
    preds = []

    # load classifier model or training data 
    if clf_type == "knn":
        train_feats, train_labels = _load_knn_train_features_from_path(path_to_image_feats)
        print("Loaded KNN train features:", train_feats.shape, "labels:", train_labels.shape)
        preds = nearest_neighbor_classify(train_feats, train_labels, test_feats, k=3)
    elif clf_type == "svm":
        if path_to_model is None or not os.path.isfile(path_to_model):
            raise FileNotFoundError("SVM model file not found: " + str(path_to_model))
        with open(path_to_model, "rb") as f:
            svms = pickle.load(f)
        for feat in test_feats:
            scores = {c: clf.decision_function([feat])[0] for c, clf in svms.items()}
            preds.append(max(scores, key=scores.get))
    else:
        raise ValueError("Unknown classifier type. Use 'KNN' or 'SVM'.")

    # evaluate performance 
    pred_arr = np.array(preds)
    true_arr = np.array(true_labels)
    acc = accuracy_score(true_arr, pred_arr)
    f1 = f1_score(true_arr, pred_arr, average="macro")

    print(f"\nOverall accuracy: {acc * 100:.2f}%")
    print(f"F1 Score (macro): {f1:.4f}")

    # display and save results
    os.makedirs("results", exist_ok=True)
    file_prefix = f"testing_{clf_type.lower()}"  # e.g., testing_knn or testing_svm
    print(f"Saving confusion matrix and metrics under prefix: {file_prefix}")

    try:
        display_results(
            true_arr.copy(),
            list(np.unique(np.concatenate([true_arr, pred_arr]))),
            pred_arr.copy(),
            output_dir="results",
            file_prefix=file_prefix
        )
    except Exception as e:
        print("Warning: display_results() failed -", str(e))

    return acc, f1, preds

# this part is just to run the testing directly
if __name__ == "__main__":
    vocab_file = "/home/bsai23021/Downloads/University/Computer Vision/assignemnt_3/scene-recognition-using-bow-and-resnet-aurangzebmalik/results/bovw_svm_new/vocab_s300_size200.pkl"
    svm_model_file = "/home/bsai23021/Downloads/University/Computer Vision/assignemnt_3/scene-recognition-using-bow-and-resnet-aurangzebmalik/results/bovw_svm_new/svm_vocab_s300_size200_model.pth"
    test_folder = "../data/test"
    image_feats_file = "/home/bsai23021/Downloads/University/Computer Vision/assignemnt_3/scene-recognition-using-bow-and-resnet-aurangzebmalik/results/bovw_svm_new/image_feats_vocab_s300_size200_train.pkl"

    # single image example (KNN)
    # testing_onOneImage("../data/test/Coast/image_0114.jpg", "KNN", path_to_model=None, path_to_BOW_VOCAB=vocab_file,path_to_image_feats=image_feats_file)

    # single image example (SVM)
    # testing_onOneImage("../data/test/Coast/image_0114.jpg", "SVM", path_to_model=svm_model_file, path_to_BOW_VOCAB=vocab_file,path_to_image_feats=image_feats_file)

    # all images example (KNN)
    testing_AllImages(test_folder, "KNN", path_to_model=None, path_to_BOW_VOCAB=vocab_file,path_to_image_feats=image_feats_file)

    # all images example (SVM)
    # testing_AllImages(test_folder, "SVM", path_to_model=svm_model_file, path_to_BOW_VOCAB=vocab_file,path_to_image_feats=image_feats_file)
