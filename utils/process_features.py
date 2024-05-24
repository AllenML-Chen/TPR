import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
import clip
from .utils import text_encode
from utils.class_template import TEMPLATE, CLASS_NAME


def inference(loader, model, device, args):
    image_vector = []
    text_vector = []
    labels_vector = []

    with torch.no_grad():
        for i, (images, label, label_names) in enumerate(tqdm(loader)):
            images = images.to(device)
            label = label.to(device)

            # Extract image features
            image_features = model.encode_image(images)       # [50, B, d]
            image_features /= image_features.norm(dim=-1, keepdim=True)  # [B, 50, d], normalized

            # class_names: [(c1_desp, c2_desp), (c1_desp, c2_desp)]: N*b
            # Extract text features
            text_features = text_encode(label_names, TEMPLATE[args.dataset], model)  # [B, d], normalized

            image_vector.append(image_features)
            text_vector.append(text_features)
            labels_vector.append(label)

    img_feat = torch.cat(image_vector, dim=0)
    txt_feat = torch.cat(text_vector, dim=0) 
    labels_vector = torch.cat(labels_vector, dim=0)
    
    img_feat = img_feat.cpu().numpy()
    txt_feat = txt_feat.cpu().numpy()
    labels_vector = labels_vector.cpu().numpy()

    feature_vector = np.stack((img_feat, txt_feat), axis=1)
    print("Image/Text features shape {}".format(feature_vector.shape))

    return feature_vector, labels_vector


def get_features(train_loader, test_seen_loader, test_unseen_loader, device, args):

    clip_model_name = args.multimodel_name
    clip.available_models()
    clip_model, preprocess = clip.load(clip_model_name)
    print('total parameter number {}'.format(sum(p.numel() for p in clip_model.parameters())))
    clip_model.to(device)
    clip_model.eval()

    # Extract paired (image, text) features
    train_X, train_y = inference(train_loader, clip_model, device, args)
    test_X_s, test_y_s = inference(test_seen_loader, clip_model, device, args)
    test_X_u, test_y_u = inference(test_unseen_loader, clip_model, device, args)

    return train_X.astype(np.float32), train_y, test_X_s.astype(np.float32), \
        test_y_s, test_X_u.astype(np.float32), test_y_u


def create_data_loaders_from_arrays(X_train, y_train, X_test_s, y_test_s, X_test_u, y_test_u, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )

    test_s = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test_s), torch.from_numpy(y_test_s)
    )
    test_seen_loader = torch.utils.data.DataLoader(
        test_s, batch_size=batch_size, shuffle=False
    )

    test_u = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test_u), torch.from_numpy(y_test_u)
    )
    test_unseen_loader = torch.utils.data.DataLoader(
        test_u, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_seen_loader, test_unseen_loader


def get_attribute_features(device, args):

    language_model_name = args.attr_model_name
    tokenizer = AutoTokenizer.from_pretrained(language_model_name)
    language_model = AutoModel.from_pretrained(language_model_name).to(device)
    language_model.eval()

    attr_list = get_attr_list(args.attr_path)

    # Tokenize and convert text to tensors
    inputs = tokenizer(attr_list, return_tensors="pt", padding=True, truncation=True, max_length=128)  
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        features = language_model(**inputs)

    # output_hidden_states = torch.mean(features.last_hidden_state, dim=1) # mean embedding
    output_hidden_states = features.last_hidden_state[:, 0, :]  # [CLS] embedding
    output_hidden_states = output_hidden_states.cpu().numpy()  # not L2 normalized
    print("Attribute features shape {}".format(output_hidden_states.shape))

    return output_hidden_states


def get_attr_list(attr_path):

    attr_list = []
    with open(attr_path, "r") as f:
        for line in f:
            attr = line.strip()
            attr_list.append(attr)
    print(attr_list[:10])
    return attr_list