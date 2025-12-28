# Name : Aurangzeb
# Roll Number : BSAI23021 
# Assignment : 3



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.cm import get_cmap



# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 
# 3-channel RGB images of shape (3xHxW)
# where H and W are expected to be atleast 224
# The images have to be loaded in to a range of [0,1]
# and then normalized using  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

# Your dataloader_resnet.py already handles all the preprocessing mentioned above , but feel free to ge through the code 
# for your knowledge

class Resnet(nn.Module):
    def __init__(self, resnet_variant='resnet18'):
        super(Resnet, self).__init__()
        
        # Load pretrained resnet backbone
        self.model = torch.hub.load('pytorch/vision:v0.10.0', resnet_variant, pretrained=True)
        
        # Remove the final fully connected layer (classifier)
        # This keeps convolutional feature extractor only (output: 512-dim feature for resnet18
        self.features = nn.Sequential(*list(self.model.children())[:-1])  
       
    def forward(self, x):
        # Forward pass through feature extractor
        x = self.features(x)   # output shape: (batch_size, 512, 1, 1) for resnet18
        x = torch.flatten(x, 1)  # flatten to (batch_size, 512)
        return x
    
    def display_tsne(features, labels, categories, title, save_path):

        print("Feature matrix shape:", features.shape)
        print("Computing t-SNE embedding...")

        # handle both string or numeric labels
        if np.issubdtype(type(labels[0]), np.integer):
            numeric_labels = np.array(labels)
        else:
            label_to_idx = {cat: i for i, cat in enumerate(categories)}
            numeric_labels = np.array([label_to_idx[lbl] for lbl in labels])

        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        # Plot
        plt.figure(figsize=(10, 8))
        cmap = get_cmap('tab10', len(categories))
        scatter = plt.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=numeric_labels,
            cmap=cmap,
            s=18,
            alpha=0.7
        )

        # Legend 
        handles = [plt.Line2D([], [], marker='o', color=cmap(i), linestyle='', label=cat)
                for i, cat in enumerate(categories)]
        plt.legend(handles=handles, title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

        print(f"✅ t-SNE visualization saved to {save_path}")
