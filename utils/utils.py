import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import random

def text_encode(classnames, templates, model):
    with torch.no_grad():
        text_feat = []
        for classname in classnames:
            # texts = [template.format(classname) for template in templates]  # format with class
            texts = classname
            texts = clip.tokenize(texts, truncate=True).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder  [1, 512]
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)   # [512]
            class_embedding /= class_embedding.norm()
            text_feat.append(class_embedding)
        text_feat = torch.stack(text_feat, dim=1).cuda()
    return text_feat.t()


def text_encode_class(classnames, templates, model):
    with torch.no_grad():
        text_feat = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts, truncate=True).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_feat.append(class_embedding)
        text_feat = torch.stack(text_feat, dim=1).cuda()
    return text_feat.t()


def zero_softmax(matrix_x, mask=None, epsilon=1e-6):
    b = matrix_x.shape[0]
    # print(mask)
    softmax_x = F.softmax(matrix_x, dim=-1)
    softmax_x = softmax_x * mask.t().repeat(b, 1)
    softmax_x = softmax_x.t() / (torch.sum(softmax_x, dim=-1) + epsilon)

    return softmax_x.t()


def compute_skl(zl, zv):
        kl_1_2 = F.kl_div(
            F.log_softmax(zl, dim=-1), F.softmax(zv, dim=-1),
            reduction='batchmean')
        kl_2_1 = F.kl_div(
            F.log_softmax(zv, dim=-1), F.softmax(zl, dim=-1),
            reduction='batchmean')
        return (kl_1_2 + kl_2_1).mean() / 2.


def generate_mask(ratio, args):
    # Calculate the number of 1s and 0s based on the ratio
    num_zeros = int(args.attr_num * ratio)
    num_ones = args.attr_num - num_zeros

    # Create a list with 1s and 0s based on the calculated counts
    data = [1] * num_ones + [0] * num_zeros

    # Shuffle the list to randomize the order of 1s and 0s
    random.shuffle(data)

    # Reshape the list into the desired shape to create the tensor
    binary_tensor = torch.tensor(data, dtype=torch.float32).view((args.attr_num, 1))

    return binary_tensor


def random_replace_ones_with_zeros(tensor, replace_ratio):

    new_tensor = tensor.clone()

    ones_indices = (new_tensor == 1).nonzero()

    num_ones_to_replace = int(len(ones_indices) * replace_ratio)

    random_indices = random.sample(range(len(ones_indices)), num_ones_to_replace)

    new_tensor[ones_indices[random_indices]] = 0

    return new_tensor




if __name__ == "__main__":
    x = torch.tensor([[0,20.0,3],[0,2,3]])
    print(x)
    mask = torch.tensor([[0],[1],[1]])
    z = zero_softmax(x, mask)
    print(z)