import argparse
import os
import numpy as np
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FGVCAircraft
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

import clip
from data.fgvc_aircraft import FGVC_Aircraft
from data.awa import AWA
from data.country import COUNTRY
from data.cub_200 import CUB
from data.flo import FLO
from data.sun import SUN
from data.stanfordcar import StanfordCars
from data.eurosat import EuroSAT
from data.dtd import DTD
from data.ucf import UCF
from data.oxfordpet import OxfordPet
from data.food import Food

from utils.process_features import get_features, create_data_loaders_from_arrays
from utils.class_template import TEMPLATE
from utils.utils import text_encode, text_encode_class
from validate import calibrate


parser = argparse.ArgumentParser(description='Universal Attribute Coordinate Training')
parser.add_argument('--data', default="./datasets/fgvc-aircraft-2013b", help='path to dataset')
parser.add_argument('--dataset', default="fgvc-aircraft", help='fgvc-aircraft|awa|country|cub|flo|sun|stanfordcar|eurosat|dtd|ucf|caltech|oxfordpet|food')
parser.add_argument('--attr_path', default="./datasets/attributes.txt", help='path to attribute set')
parser.add_argument("--attr_model_name", type=str, default="bert-base-uncased", help="Pretrained BERT model name")
parser.add_argument("--multimodel_name", type=str, default="ViT-B/32", help="Pretrained CLIP model name")

parser.add_argument('--b', default=512, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. (default: None)')


os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def main():

    args = parser.parse_args()


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the image data transformations 
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),            
    ])

    Dataset = {'fgvc-aircraft': FGVC_Aircraft, 'awa': AWA, 'country': COUNTRY,
               'cub': CUB, 'flo': FLO, 'sun': SUN, 'stanfordcar': StanfordCars, 'eurosat': EuroSAT, 'dtd': DTD, 
               'ucf': UCF, 'caltech': CalTech, 'oxfordpet': OxfordPet, 'food': Food}[args.dataset]

     # Load the dataset  (image, label, class name)
    train_dataset = Dataset(args.data, split='train', transform=transform)
    test_seen_dataset = Dataset(args.data, split='test_seen', transform=transform)
    test_unseen_dataset = Dataset(args.data, split='test_unseen', transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    test_seen_loader = DataLoader(test_seen_dataset, batch_size=args.b, shuffle=False)
    test_unseen_loader = DataLoader(test_unseen_dataset, batch_size=args.b, shuffle=False)

    # Precompute features
    feature_file = "./features/features_{}_gzsl.p".format(args.dataset)
    if not os.path.exists(feature_file):
        print("### Creating features from pre-trained model ###")
        train_X, train_y, test_X_s, test_y_s, test_X_u, test_y_u = get_features(
            train_loader, test_seen_loader, test_unseen_loader, device, args
        )
        pickle.dump(
            (train_X, train_y, test_X_s, test_y_s, test_X_u, test_y_u), open(feature_file, "wb"), protocol=4
        )
    else:
        print("### Loading features ###")
        train_X, train_y, test_X_s, test_y_s, test_X_u, test_y_u = pickle.load(open(feature_file, "rb"))

        num_train_classes = len(np.unique(train_y))
        num_test_seen_classes = len(np.unique(test_y_s))
        num_test_unseen_classes = len(np.unique(test_y_u))
        print("Number of train classes:", num_train_classes)
        print("Number of test seen classes:", num_test_seen_classes)
        print("Number of test unseen classes:", num_test_unseen_classes)
        print("Train image features shape {}, Test seen image features shape {}, Test unseen image features shape {}"\
              .format(train_X.shape, test_X_s.shape, test_X_u.shape))

    # Create data loaders [(iamge features, label features), labels]
    train_loader, test_seen_loader, test_unseen_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X_s, test_y_s, test_X_u, test_y_u, args.b 
    )


    # ------------------------------------------ CLIP Zero-Shot Evaluation ------------------------------------------

    clip_model, preprocess = clip.load(args.multimodel_name)
    clip_model.to(device)
    clip_model.eval()

    clip_text_features_class = text_encode_class(train_dataset.classes, TEMPLATE[args.dataset], clip_model)
    clip_text_features_class = clip_text_features_class.to(torch.float32)
    simi_s_clip = predict_clip(test_seen_loader, clip_text_features_class, device)
    simi_u_clip = predict_clip(test_unseen_loader, clip_text_features_class, device)
    seen_idx = train_dataset.seen_idx
    
    res_clip = calibrate(simi_s_clip, simi_u_clip, test_seen_dataset.labels,
              test_unseen_dataset.labels, seen_idx)
    print("CLIP results with class template: " + res_clip)



def predict_clip(test_loader, clip_text_features, device):
    with torch.no_grad():
        simis = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            clip_img_features = inputs[:, 0, :]
            simi = torch.matmul(clip_img_features, clip_text_features.t())  # [B, num_unseen_classes]
            simis.append(simi)

    simis = torch.cat(simis, dim=0)
    return simis


if __name__ == "__main__":
    main()