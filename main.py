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
from data.awa import AWA
from models.dis_text import TextRep
from losses.loss import SupConLoss
from utils.process_features import get_features, create_data_loaders_from_arrays, get_attribute_features
from utils.class_template import TEMPLATE, CLASS_NAME
from utils.utils import text_encode, text_encode_class
from validate import calibrate


parser = argparse.ArgumentParser(description='TPR Training')
parser.add_argument('--data', default="./datasets/AwA2", help='path to dataset')
parser.add_argument('--dataset', default="awa", help='path to dataset')
parser.add_argument('--attr_path', default="./datasets/attributes.txt", help='path to attribute set')
parser.add_argument("--attr_model_name", type=str, default="bert-base-uncased", help="Pretrained BERT model name")
parser.add_argument("--multimodel_name", type=str, default="ViT-B/32", help="Pretrained CLIP model name")

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--b', default=512, type=int,
                    help='mini-batch size, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay',
                    dest='weight_decay')
parser.add_argument('--attribute_list_path', default='', type=str, metavar='PATH',
                    help='path to attribute list')

# model specific configs:
parser.add_argument('--attr_num', default=5996, type=int,
                    help='number of attributes, it depends on the size of predefined attribute set')
parser.add_argument('--num_register_tokens', default=300, type=int,
                    help='number of attribute register tokens')
parser.add_argument('--attr_dim', default=768, type=int,
                    help='feature dimension of extracted attribute embeddings')
parser.add_argument('--feat_dim', default=512, type=int,
                    help='feature dimension of extracted input text and image embeddings')
parser.add_argument('--output_dim', default=512, type=int,
                    help='feature dimension of embedding space')

parser.add_argument('--beta', default=1e-3, type=float,
                    help=' loss weight')
parser.add_argument('--eta', default=0.3, type=float,
                    help='loss weight')
parser.add_argument('--temp', default=0.04, type=float,
                    help='softmax temperature')

parser.add_argument('--seed', default=7, type=int,
                    help='seed for initializing training.')


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():

    args = parser.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the data transformations 
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),           
    ])

    # Load the CUB dataset  (image, label, class name)
    train_dataset = AWA(args.data, split='train', transform=transform)
    test_seen_dataset = AWA(args.data, split='test_seen', transform=transform)
    test_unseen_dataset = AWA(args.data, split='test_unseen', transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)
    test_seen_loader = DataLoader(test_seen_dataset, batch_size=args.b, shuffle=False)
    test_unseen_loader = DataLoader(test_unseen_dataset, batch_size=args.b, shuffle=False)

    # Obtain attribute features (N_attr, attr_dim)
    if not os.path.exists("./features/attribute_features.p"):
        attr_feat = get_attribute_features(device, args).astype(np.float32)
        pickle.dump(
            attr_feat, open("./features/attribute_features.p", "wb"), protocol=4
        )

    model = TextRep(device, args).to(device)

    # Define optimizer and criterion 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = SupConLoss(temperature=args.temp)

    # Precompute features
    if not os.path.exists("./features/features_awa_gzsl.p"):
        print("### Creating features from pre-trained model ###")
        train_X, train_y, test_X_s, test_y_s, test_X_u, test_y_u = get_features(
            train_loader, test_seen_loader, test_unseen_loader, device, args
        )
        pickle.dump(
            (train_X, train_y, test_X_s, test_y_s, test_X_u, test_y_u), open("./features/features_awa_gzsl.p", "wb"), protocol=4
        )
    else:
        print("### Loading features ###")
        train_X, train_y, test_X_s, test_y_s, test_X_u, test_y_u = pickle.load(open("./features/features_awa_gzsl.p", "rb"))

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

    classes = train_dataset.class_texts
    clip_model, preprocess = clip.load(args.multimodel_name)
    clip_model.to(device)
    clip_model.eval()
    clip_text_features = text_encode(classes, TEMPLATE[args.dataset], clip_model)
    clip_text_features = clip_text_features.to(torch.float32)
    print("clip text features shape {}, norm {}".format(clip_text_features.shape, clip_text_features.norm(dim=1)))

    clip_text_features_class = text_encode_class(train_dataset.classes, TEMPLATE[args.dataset], clip_model)
    clip_text_features_class = clip_text_features_class.to(torch.float32)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            # Note inputs: [image_features, label_features] with size [B, 2, d]
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            img_attr_rep, img_feat, txt_attr_rep, txt_feat = model(inputs)    #  [B, N_attr]

            # L2 normalized the features
            attr_rep = torch.stack((img_attr_rep, txt_attr_rep), dim=1)   # [B, 2, N_attr]
            attr_rep = F.normalize(attr_rep, p=2, dim=2)

            feat = torch.stack((img_feat, txt_feat), dim=1)   # [B, 2, d]
            feat = F.normalize(feat, p=2, dim=2)

            # Compute the SupCon loss
            sc_loss = args.eta * criterion(feat, labels) + criterion(attr_rep, labels)

            # Compute text distinct loss
            txt_attr_rep_, txt_feat_ = model.get_text_representation(clip_text_features)  # [N_c, d]

            noise = 0.5*torch.randn(50, args.feat_dim).to(device)
            topo_loss = pearson_loss(txt_attr_rep_, clip_text_features + noise)

            loss = sc_loss + args.beta * topo_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {running_loss / len(train_loader)}, LR: {optimizer.param_groups[0]['lr']}")
    print("Training completed.")

    # ------------------------------------------ Evaluation ------------------------------------------
    model.eval()

    simi_s = predict(test_seen_loader, model, clip_text_features, device)
    simi_u = predict(test_unseen_loader, model, clip_text_features, device)
    seen_idx = train_dataset.seen_idx
    
    res = calibrate(simi_s, simi_u, test_seen_dataset.labels,
              test_unseen_dataset.labels, seen_idx)
    print(res)

    # ------------------------------------------ CLIP Zero-shot ------------------------------------------
    simi_s_clip = predict_clip(test_seen_loader, clip_text_features_class, device)
    simi_u_clip = predict_clip(test_unseen_loader, clip_text_features_class, device)
    seen_idx = train_dataset.seen_idx
    
    res_clip = calibrate(simi_s_clip, simi_u_clip, test_seen_dataset.labels,
              test_unseen_dataset.labels, seen_idx)
    print("CLIP results with class template: " + res_clip)


def pearson_loss(ref_norm_feat, cur_norm_feat):
    ref_norm_feat = F.normalize(ref_norm_feat, dim=1)
    cur_norm_feat = F.normalize(cur_norm_feat, dim=1)
    ref_rank = torch.mm(ref_norm_feat, (ref_norm_feat.transpose(1, 0)))
    cur_rank = torch.mm(cur_norm_feat, (cur_norm_feat.transpose(1, 0)))

    x, y = ref_rank.shape

    mref = torch.mean(ref_rank, 1)
    mcur = torch.mean(cur_rank, 1)

    refm = ref_rank - mref.repeat(y).reshape(y, x).transpose(1, 0)
    curm = cur_rank - mcur.repeat(y).reshape(y, x).transpose(1, 0)
    refm = refm.to(curm.device)
    r_num = torch.sum(refm * curm, 1)
    r_den = torch.sqrt(torch.sum(torch.pow(refm, 2), 1) * torch.sum(torch.pow(curm, 2), 1))
    r = 1 - (r_num / r_den)
    cor = torch.mean(r)
    return cor


def predict(test_loader, model, text_feature, device):
    with torch.no_grad():
        # Precompute unseen class features
        txt_attr_rep, txt_feat = model.get_text_representation(text_feature)  # [num_unseen_classes, d]
        txt_feat = F.normalize(txt_feat, p=2, dim=1)
        simis = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            img_attr_rep, img_feat = model.get_image_representation(inputs[:, 0, :])  # [B, d]
            img_feat = F.normalize(img_feat, p=2, dim=1)

            # weighted prediction
            simi = torch.matmul(img_feat, txt_feat.t())
            simis.append(simi)
    simis = torch.cat(simis, dim=0)
    return simis


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