import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def _CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(
            inputs, size=(ht, wt), mode="bilinear", align_corners=True
        )

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)
    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(
        temp_inputs, temp_target
    )
    return CE_loss


def BCE_Loss(inputs, target, num_classes, sig=True):
    temp_inputs = inputs.squeeze(1).contiguous().view(-1)
    temp_target = target.view(-1)
    valid = temp_target != num_classes
    temp_inputs = temp_inputs[valid]
    temp_target = temp_target[valid]
    weight = torch.zeros(temp_target.shape).cuda()
    with torch.no_grad():
        fg = torch.sum(temp_target)
        bg = torch.sum(1 - temp_target)
        weight[temp_target == 1] = bg / (bg + fg)
        weight[temp_target == 0] = fg / (bg + fg) * 1.1
    if sig:
        BCE_Loss = F.binary_cross_entropy_with_logits(
            temp_inputs, temp_target, weight=weight
        )
    else:
        BCE_Loss = F.binary_cross_entropy(
            temp_inputs, temp_target, weight=weight, reduction="mean"
        )
    return BCE_Loss


def dice_loss_bi(input, target, num_classes=2):
    smooth = 1e-5
    n = input.size(0)
    target_s = target.unsqueeze(1)
    input_s = torch.sigmoid(input)
    input_s = input_s * (target_s != num_classes)
    target_s[target_s == num_classes] = 0
    iflat = input_s.view(n, -1)
    tflat = target_s.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    s_in = iflat.sum(1)
    s_tg = tflat.sum(1)
    loss = 1 - ((2.0 * intersection + smooth) / (s_in + s_tg + smooth))
    # print(intersection.mean(), s_in.mean(), s_tg.mean())
    return loss.mean()


def CE_Loss(inputs, target, weight, cls_weights, num_classes=2):
    # print(inputs.size())
    temp_inputs = inputs.squeeze(1).contiguous().view(-1)
    temp_target = target.view(-1)
    valid = temp_target != num_classes
    temp_inputs = temp_inputs[valid]
    temp_target = temp_target[valid]
    # temp_weight = weight.view(-1)[valid]
    BCE_Loss = F.binary_cross_entropy_with_logits(
        temp_inputs, temp_target, reduction="none"
    )  #####
    # BCE_Loss = BCE_Loss * temp_weight
    return BCE_Loss.mean()


def Focal_Loss(inputs, target, cls_weights, num_classes=2, alpha=0.5, gamma=2):
    temp_inputs = inputs.squeeze(1).contiguous().view(-1)
    temp_target = target.view(-1)
    valid = temp_target != num_classes
    temp_inputs = temp_inputs[valid]
    temp_target = temp_target[valid]
    logpt = F.binary_cross_entropy_with_logits(
        temp_inputs, temp_target, reduction="none"
    )
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(
            inputs, size=(ht, wt), mode="bilinear", align_corners=True
        )

    temp_inputs = torch.softmax(
        inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1
    )
    temp_target = target.view(n, -1, ct)

    #   calculate dice loss
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    mid_inputs = torch.squeeze(1 - temp_target[..., -1]).unsqueeze(2).repeat(1, 1, 2)
    fp = torch.sum(temp_inputs * mid_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp
    score = ((1 + beta**2) * tp + smooth) / (
        (1 + beta**2) * tp + beta**2 * fn + fp + smooth
    )
    dice_loss = 1 - torch.mean(score[1:])
    return dice_loss


# add lovasz_softmax loss
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(inputs, labels, classes="present", per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """

    probas = torch.softmax(inputs, 1)
    if per_image:
        loss = mean(
            lovasz_softmax_flat(
                *flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                classes=classes
            )
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = lovasz_softmax_flat(
            *flatten_probas(probas, labels, ignore), classes=classes
        )
    return loss


def lovasz_softmax_flat(probas, labels, classes="present"):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes is "present" and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)
    # return losses[1]


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def mean(l, empty=0):
    l = iter(l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


## binary classification lovasz


def _lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(
            _lovasz_hinge_flat(
                *_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: [P] Logits at each prediction (between -infinity and +infinity)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def weights_init(net, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and classname.find("Conv") != -1:
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print("initialize network with %s type" % init_type)
    net.apply(init_func)


def CEL(inputs, target, num_classes):
    eps = 1e-6
    pred = inputs.squeeze(1).contiguous().sigmoid()
    valid = target != num_classes
    pred_t = pred[valid]
    target_t = target[valid]
    intersection = pred_t * target_t
    numerator = (pred_t - intersection).sum() + (target_t - intersection).sum()
    denominator = pred_t.sum() + target_t.sum()
    return numerator / (denominator + eps)


class CBL(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CBL, self).__init__()
        self.laplacian_kernel = (
            torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32)
            .reshape(1, 1, 3, 3)
            .requires_grad_(False)
            .type(torch.cuda.FloatTensor)
        )

    def forward(self, edges, labels):
        eg = torch.sigmoid(edges).detach()
        eg[eg > 0.5] = 1
        eg[eg <= 0.5] = 0
        CBL_loss = BCE_Loss(labels, eg, 2, True)
        return CBL_loss


def bdrloss(prediction, label, radius):
    """
    The boundary tracing loss that handles the confusing pixels.
    """
    prediction = torch.sigmoid(prediction)
    label = label.unsqueeze(1)
    filt = torch.ones(1, 1, 2 * radius + 1, 2 * radius + 1)
    filt.requires_grad = False
    filt = filt.cuda()
    filt0 = torch.ones(label.shape)
    filt0.requires_grad = False
    filt0 = filt0.cuda()
    filt0[label == 2] = 0
    prediction = prediction * filt0
    label = label * filt0

    bdr_pred = prediction * label
    pred_bdr_sum = label * F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)

    texture_mask = F.conv2d(label.float(), filt, bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(
        prediction * (1 - label) * mask, filt, bias=None, stride=1, padding=radius
    )

    softmax_map = torch.clamp(
        pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10
    )
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return cost.mean()


def textureloss(prediction, label, mask_radius):
    """
    The texture suppression loss that smooths the texture regions.
    """
    prediction = torch.sigmoid(prediction)
    label = label.unsqueeze(1)
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.cuda()
    filt2 = torch.ones(1, 1, 2 * mask_radius + 1, 2 * mask_radius + 1)
    filt2.requires_grad = False
    filt2 = filt2.cuda()
    filt0 = torch.ones(label.shape)
    filt0.requires_grad = False
    filt0 = filt0.cuda()
    filt0[label == 2] = 0
    prediction = prediction * filt0
    label = label * filt0

    pred_sums = F.conv2d(prediction.float(), filt1, bias=None, stride=1, padding=1)
    label_sums = F.conv2d(
        label.float(), filt2, bias=None, stride=1, padding=mask_radius
    )

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1 - pred_sums / 9, 1e-10, 1 - 1e-10))
    loss[mask == 0] = 0

    return loss.mean()
