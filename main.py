# before running this make visualize = false in get_bag_of_sift function
# it is made inside the result folder (can be either of any 4 pipeline) named visualization_train 




from DataLoader_Resnet import CustomImageDataset
from Resnet_Backbone import Resnet
from BOVW import build_vocabulary, get_bags_of_sifts
from KNN import nearest_neighbor_classify
from SVM import svm_classify
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle
import os
from utils import get_image_paths , display_results
import argparse


def map_feature_choice(arg_feature):
    a = arg_feature.lower()
    if a == 'resnet':
        return 'resnet'
    if a in ['bovw', 'bag_of_sift', 'bag-of-sift', 'bagofsift', 'bag of sift']:
        return 'bag of sift'
    return arg_feature


def map_classifier_choice(arg_clf):
    a = arg_clf.lower()
    if a in ['knn', 'nearest', 'nearest_neighbor', 'nearest-neighbor']:
        return 'nearest neighbor'
    if a in ['svm', 'support_vector_machine', 'support-vector-machine', 'support vector machine']:
        return 'support vector machine'
    return arg_clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scene recognition project with args')
    parser.add_argument('-i', '--input', default='../data/', help='input data path')
    parser.add_argument('-o', '--output', default='results', help='output path to save results')
    parser.add_argument('-c', '--classifier', choices=['knn', 'svm'], default='svm', help='choose knn or svm')
    parser.add_argument('-f', '--feature', choices=['resnet', 'bovw', 'bag_of_sift', 'bag-of-sift', 'bagofsift', 'bag of sift'], default='bovw', help='choose resnet or bovw')
    args = parser.parse_args()

    # setting up data folders and model options
    data_path = args.input
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    FEATURE = map_feature_choice(args.feature)
    CLASSIFIER = map_classifier_choice(args.classifier)

    print('Parsed arguments:')
    print(f'  input data path: {data_path}')
    print(f'  output folder:   {OUTPUT_DIR}')
    print(f'  feature:         {FEATURE}')
    print(f'  classifier:      {CLASSIFIER}')

    # getting all categories (scene types)
    categories = np.array([
        'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
        'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
        'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'
    ])

    print('getting image paths...')
    train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path, categories)

    # mapping label names to numbers for easier training
    categories = sorted(list(set(train_labels)))
    class_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    print("using", FEATURE, "features...")

    # using resnet to make image features
    if FEATURE == 'resnet':
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        train_dataset = CustomImageDataset(train_image_paths, train_labels, class_to_idx, transform=transform)
        test_dataset  = CustomImageDataset(test_image_paths, test_labels, class_to_idx, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # checking gpu and loading resnet model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet_model = Resnet('resnet18').to(device)
        resnet_model.eval()

        # this makes features from images using resnet
        def extract_features(loader):
            feats_list = []
            labels_list = []
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    feats = resnet_model(images)
                    feats_list.append(feats.cpu().numpy())
                    labels_list.extend(labels)
            return np.vstack(feats_list), np.array(labels_list)

        resnet_train_path = os.path.join(OUTPUT_DIR, 'resnet_train_feats.pkl')
        resnet_test_path = os.path.join(OUTPUT_DIR, 'resnet_test_feats.pkl')

        # saving or loading features so we don’t do it again
        if not os.path.isfile(resnet_train_path):
            train_feats, train_labels_arr = extract_features(train_loader)
            with open(resnet_train_path, 'wb') as f:
                pickle.dump((train_feats, train_labels_arr), f)
        else:
            with open(resnet_train_path, 'rb') as f:
                train_feats, train_labels_arr = pickle.load(f)

        if not os.path.isfile(resnet_test_path):
            test_feats, test_labels_arr = extract_features(test_loader)
            with open(resnet_test_path, 'wb') as f:
                pickle.dump((test_feats, test_labels_arr), f)
        else:
            with open(resnet_test_path, 'rb') as f:
                test_feats, test_labels_arr = pickle.load(f)

        # putting features and labels ready for training
        train_image_feats = train_feats
        test_image_feats = test_feats
        train_labels = train_labels_arr
        test_labels = test_labels_arr

        # this part draws t-sne graph to see how resnet sees images
        Resnet.display_tsne(train_image_feats, train_labels, categories,
                 title="ResNet Feature Visualization",
                 save_path=os.path.join(OUTPUT_DIR, "resnet_tsne.png"))

        print('training classifier...')
        if CLASSIFIER == 'nearest neighbor':
            predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
        elif CLASSIFIER == 'support vector machine':
            svms, predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
            with open(os.path.join(OUTPUT_DIR, f"svm_{FEATURE}_model.pth"), "wb") as f:
                pickle.dump(svms, f)
        else:
            predicted_categories = np.random.choice(categories, size=len(test_labels))

        # showing results like confusion matrix and accuracy
        print(f"\n--- results for {CLASSIFIER} + {FEATURE} ---")
        file_prefix = f"{CLASSIFIER.replace(' ', '_')}_resnet"
        predicted_categories = np.array(predicted_categories)
        display_results(test_labels, categories, predicted_categories, output_dir=OUTPUT_DIR, file_prefix=file_prefix)

    # using bag of sift (bovw) to make image features
    elif FEATURE == 'bag of sift':
        num_descriptors_list = [300]
        vocab_sizes = [200]

        # looping through vocab and feature setups
        for num_desc in num_descriptors_list:
            for vocab_size in vocab_sizes:
                print(f"\n--- running bovw: s={num_desc}, vocab_size={vocab_size}, classifier={CLASSIFIER} ---")
                vocab_file = os.path.join(OUTPUT_DIR, f'vocab_s{num_desc}_size{vocab_size}.pkl')

                # building vocab with kmeans
                if not os.path.isfile(vocab_file):
                    print(f'making vocab {vocab_file}...')
                    vocab = build_vocabulary(train_image_paths, vocab_size, num_descriptors=num_desc)
                    with open(vocab_file, 'wb') as f:
                        pickle.dump(vocab, f)
                else:
                    print(f'loading vocab {vocab_file}...')
                    with open(vocab_file, 'rb') as f:
                        vocab = pickle.load(f)

                # making histogram features using vocab
                print(f'getting bovw features...')
                train_image_feats = get_bags_of_sifts(
                train_image_paths, 
                vocab_path=vocab_file, 
                num_descriptors=num_desc, 
                labels=train_labels, 
                visualize=True, 
                vis_dir=os.path.join(OUTPUT_DIR, "visualizations_train")
                )

                test_image_feats = get_bags_of_sifts(
                    test_image_paths, 
                    vocab_path=vocab_file, 
                    num_descriptors=num_desc, 
                    labels=test_labels, 
                    visualize=False  
                )


                vname = os.path.basename(vocab_file).replace(".pkl", "")
                train_save = os.path.join(OUTPUT_DIR, f"image_feats_{vname}_train.pkl")
                test_save  = os.path.join(OUTPUT_DIR, f"image_feats_{vname}_test.pkl")

                with open(train_save, "wb") as f:
                    pickle.dump({"features": train_image_feats, "labels": train_labels}, f)
                print("saved train features")

                with open(test_save, "wb") as f:
                    pickle.dump({"features": test_image_feats, "labels": test_labels}, f)
                print("saved test features")

                # training the classifier on bovw features
                print('training classifier...')
                if CLASSIFIER == 'nearest neighbor':
                    predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
                elif CLASSIFIER == 'support vector machine':
                    model_save_path = os.path.join(OUTPUT_DIR, f"svm_{vname}_model.pth")
                    svms, predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
                    with open(model_save_path, "wb") as f:
                        pickle.dump(svms, f)
                    print("saved svm model")
                else:
                    predicted_categories = np.random.choice(categories, size=len(test_labels))

                # t-sne graph to show bovw feature pattern
                Resnet.display_tsne(train_image_feats, train_labels, categories,
                                 title=f"BoVW Feature Visualization ({vname})",
                                 save_path=os.path.join(OUTPUT_DIR, f"bovw_tsne_{vname}.png"))

                # showing final results for this setup
                print(f"\n--- results for {CLASSIFIER} + {vname} ---")
                file_prefix = f"{CLASSIFIER.replace(' ', '_')}_{vname}"
                predicted_categories = np.array(predicted_categories)
                display_results(test_labels, categories, predicted_categories, output_dir=OUTPUT_DIR, file_prefix=file_prefix)
