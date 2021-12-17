import torch.nn as nn

def replace_max_pool(fs, c = 4, m = nn.MaxPool2d(3, 1, 1)):
    index = -1
    count = 0
    for idx, layer in enumerate(fs):
        if isinstance(layer, nn.MaxPool2d):
            count += 1
        if count == c:
            index = idx
            break

    assert index > 0

    fs[index] = m
    return fs