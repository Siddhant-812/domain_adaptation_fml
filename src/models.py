"""Backbone construction and update-scope control."""
import torch
import torch.nn as nn
import torchvision

from data import to_input


def build_model(arch, num_classes, pretrained=True):
    if arch == "resnet18":
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    elif arch == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    else:
        raise ValueError(arch)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _is_bn(m):
    return isinstance(m, nn.modules.batchnorm._BatchNorm)


def configure_scope(model, scope):
    """Freeze everything, then unfreeze the parameters of the chosen scope.

    full      : all parameters
    lastblock : layer4 + classifier
    bn        : BatchNorm affine (gamma, beta) only  -- TENT-style
    head      : classifier only (encoder frozen)
    none      : nothing (fixed source model)
    """
    for p in model.parameters():
        p.requires_grad_(False)
    if scope == "full":
        params = list(model.parameters())
    elif scope == "lastblock":
        params = list(model.layer4.parameters()) + list(model.fc.parameters())
    elif scope == "bn":
        params = [p for m in model.modules() if _is_bn(m) for p in (m.weight, m.bias)]
    elif scope == "head":
        params = list(model.fc.parameters())
    elif scope == "none":
        params = []
    else:
        raise ValueError(scope)
    for p in params:
        p.requires_grad_(True)
    return params


def set_train_mode(model, scope):
    """BatchNorm statistics are updated only for layers inside the scope."""
    model.eval()
    if scope == "full":
        model.train()
    elif scope == "lastblock":
        model.layer4.train()
    elif scope == "bn":
        for m in model.modules():
            if _is_bn(m):
                m.train()


@torch.no_grad()
def predict(model, x_uint8, idx, amp, batch=256):
    """Return softmax probabilities for x[idx] in eval mode."""
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        xb = to_input(x_uint8[idx[i:i + batch]])
        with amp:
            out.append(torch.softmax(model(xb).float(), 1))
    return torch.cat(out) if out else torch.empty(0)


@torch.no_grad()
def evaluate(model, dom, amp):
    """Accuracy and collapse index (largest predicted-class share) on the test split."""
    idx = dom["test_idx"]
    p = predict(model, dom["x"], idx, amp)
    pred = p.argmax(1)
    acc = (pred == dom["y"][idx]).float().mean().item()
    share = torch.bincount(pred, minlength=p.shape[1]).max().item() / len(pred)
    return acc, share
