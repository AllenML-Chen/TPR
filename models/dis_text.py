import pickle
import os.path as osp
from collections import OrderedDict
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F



def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, value) 

    return output, attn_weights


class ImageToAttributeTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, attr_dim):
        super(ImageToAttributeTransformer, self).__init__()

        # Layer Normalization
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, img_features, attribute_features):
        x = img_features

        img_features, _ = scaled_dot_product_attention(img_features, attribute_features, attribute_features)
        img_features = img_features + x
        img_features = self.norm1(img_features)  
        
        img_features, _ = scaled_dot_product_attention(img_features, img_features, img_features)
        img_features = img_features + x
        img_features = self.norm2(img_features)

        img_attr_simi = torch.matmul(img_features, attribute_features.t())

        return img_attr_simi, img_features


class TextToAttributeTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, attr_dim):
        super(TextToAttributeTransformer, self).__init__()
        
    def forward(self, text_features, attribute_features):
        x = text_features

        text_features, _ = scaled_dot_product_attention(text_features, attribute_features, attribute_features)
        
        text_features = text_features + x
        txt_attr_simi = torch.matmul(text_features, attribute_features.t())

        return txt_attr_simi, text_features


class TextRep(nn.Module):
    def __init__(self, device, args):
        super(TextRep, self).__init__()
        
        self.device = device
        self.input_dim = args.feat_dim
        self.output_dim = args.output_dim
        self.attr_dim = args.attr_dim
        self.num_register_tokens = args.num_register_tokens

        # Project images to attributes
        self.img_to_attr = self.build_image_to_attributes_transformer()

        # Project text to attributes
        self.txt_to_attr = self.build_text_to_attributes_transformer()

        # attribute register tokens
        self.attr_register_tokens = nn.Parameter(
            torch.randn(self.num_register_tokens, self.attr_dim)
        ).to(self.device)

        # Obtain attribute features
        self.attr_feat_ = self.load_attribute_features()
        self.attr_feat = torch.cat([self.attr_feat_, self.attr_register_tokens], dim=0)

        self.projection_layer_v = nn.Linear(self.input_dim, self.output_dim)
        self.projection_layer_l = self.projection_layer_v
        self.proj_v = nn.Linear(self.attr_dim, self.output_dim)
        self.proj_l = nn.Linear(self.attr_dim, self.output_dim)

    # build model
    def build_image_to_attributes_transformer(self):
        return ImageToAttributeTransformer(self.input_dim, self.output_dim, self.attr_dim)

    def build_text_to_attributes_transformer(self):
        return TextToAttributeTransformer(self.input_dim, self.output_dim, self.attr_dim)
    
    def load_attribute_features(self):
        attr_feat = pickle.load(open("./features/attribute_features.p", "rb"))
        attr_feat = torch.from_numpy(attr_feat)
        return attr_feat.to(self.device)

    def get_text_representation(self, text_features):
        attribute_features = self.proj_l(self.attr_feat)
        text_features = self.projection_layer_l(text_features)
        txt_attr_rep, txt_feat = self.txt_to_attr(text_features, attribute_features)
        return txt_attr_rep, txt_feat

    def get_image_representation(self, image_features):
        attribute_features = self.proj_v(self.attr_feat)
        image_features = self.projection_layer_v(image_features)
        img_attr_rep, img_feat = self.img_to_attr(image_features, attribute_features)
        return img_attr_rep, img_feat

    def forward(self, inputs):
        image_features, text_features = inputs[:, 0, :], inputs[:, 1, :]

        image_features = self.projection_layer_v(image_features)
        text_features = self.projection_layer_l(text_features)
        attribute_features_v = self.proj_v(self.attr_feat)
        attribute_features_l = self.proj_l(self.attr_feat)

        # Project image features into attribute space
        img_attr_rep, img_feat = self.img_to_attr(image_features, attribute_features_v)  # [B, N_attr], [B, d]
        
        # Project text features into attribute space
        txt_attr_rep, txt_feat = self.txt_to_attr(text_features, attribute_features_l)   # [B, N_attr], [B, d]

        return img_attr_rep, img_feat, txt_attr_rep, txt_feat