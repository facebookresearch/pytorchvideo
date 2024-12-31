import os
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import os
from torchvision.ops import nms


try:
    from utils.syncbn import SyncBN

    batch_norm = SyncBN  # nn.BatchNorm2d
except ImportError:
    batch_norm = nn.BatchNorm2d


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                after_bn = batch_norm(filters)
                modules.add_module('batch_norm_%d' % i, after_bn)
                # BN is uniformly initialized by default in pytorch 1.0.1.
                # In pytorch>1.2.0, BN weights are initialized with constant 1,
                # but we find with the uniform initialization the model converges faster.
                nn.init.uniform_(after_bn.weight)
                nn.init.zeros_(after_bn.bias)
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            nC = int(module_def['classes'])  # number of classes
            img_size = (int(hyperparams['width']), int(hyperparams['height']))
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, nC, int(hyperparams['nID']),
                                   int(hyperparams['embedding_dim']), img_size, yolo_layer_count)
            modules.add_module('yolo_%d' % i, yolo_layer)
            yolo_layer_count += 1

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC, nID, nE, img_size, yolo_layer):
        super(YOLOLayer, self).__init__()
        self.layer = yolo_layer
        nA = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.nID = nID  # number of identities
        self.img_size = 0
        self.emb_dim = nE
        self.shift = [1, 3, 5]

        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.SoftmaxLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.s_c = nn.Parameter(-4.15 * torch.ones(1))  # -4.15
        self.s_r = nn.Parameter(-4.85 * torch.ones(1))  # -4.85
        self.s_id = nn.Parameter(-2.3 * torch.ones(1))  # -2.3

        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1) if self.nID > 1 else 1

    def forward(self, p_cat, img_size, targets=None, classifier=None, test_emb=False):
        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]
        nB, nGh, nGw = p.shape[0], p.shape[-2], p.shape[-1]

        if self.img_size != img_size:
            create_grids(self, img_size, nGh, nGw)

            if p.is_cuda:
                self.grid_xy = self.grid_xy.cuda()
                self.anchor_wh = self.anchor_wh.cuda()

        p = p.view(nB, self.nA, self.nC + 5, nGh, nGw).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        p_emb = p_emb.permute(0, 2, 3, 1).contiguous()
        p_box = p[..., :4]
        p_conf = p[..., 4:6].permute(0, 4, 1, 2, 3)  # Conf

        # Training
        if targets is not None:
            if test_emb:
                tconf, tbox, tids = build_targets_max(targets, self.anchor_vec.cuda(), self.nA, self.nC, nGh, nGw)
            else:
                tconf, tbox, tids = build_targets_thres(targets, self.anchor_vec.cuda(), self.nA, self.nC, nGh, nGw)
            tconf, tbox, tids = tconf.cuda(), tbox.cuda(), tids.cuda()
            mask = tconf > 0

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            nP = torch.ones_like(mask).sum().float()
            if nM > 0:
                lbox = self.SmoothL1Loss(p_box[mask], tbox[mask])
            else:
                FT = torch.cuda.FloatTensor if p_conf.is_cuda else torch.FloatTensor
                lbox, lconf = FT([0]), FT([0])
            lconf = self.SoftmaxLoss(p_conf, tconf)
            lid = torch.Tensor(1).fill_(0).squeeze().cuda()
            emb_mask, _ = mask.max(1)

            # For convenience we use max(1) to decide the id, TODO: more reseanable strategy
            tids, _ = tids.max(1)
            tids = tids[emb_mask]
            embedding = p_emb[emb_mask].contiguous()
            embedding = self.emb_scale * F.normalize(embedding)
            nI = emb_mask.sum().float()

            if test_emb:
                if np.prod(embedding.shape) == 0 or np.prod(tids.shape) == 0:
                    return torch.zeros(0, self.emb_dim + 1).cuda()
                emb_and_gt = torch.cat([embedding, tids.float()], dim=1)
                return emb_and_gt

            if len(embedding) > 1:
                logits = classifier(embedding).contiguous()
                lid = self.IDLoss(logits, tids.squeeze())

            # Sum loss components
            loss = torch.exp(-self.s_r) * lbox + torch.exp(-self.s_c) * lconf + torch.exp(-self.s_id) * lid + \
                   (self.s_r + self.s_c + self.s_id)
            loss *= 0.5

            return loss, loss.item(), lbox.item(), lconf.item(), lid.item(), nT

        else:
            p_conf = torch.softmax(p_conf, dim=1)[:, 1, ...].unsqueeze(-1)
            p_emb = F.normalize(p_emb.unsqueeze(1).repeat(1, self.nA, 1, 1, 1).contiguous(), dim=-1)
            # p_emb_up = F.normalize(shift_tensor_vertically(p_emb, -self.shift[self.layer]), dim=-1)
            # p_emb_down = F.normalize(shift_tensor_vertically(p_emb, self.shift[self.layer]), dim=-1)
            p_cls = torch.zeros(nB, self.nA, nGh, nGw, 1).cuda()  # Temp
            p = torch.cat([p_box, p_conf, p_cls, p_emb], dim=-1)
            # p = torch.cat([p_box, p_conf, p_cls, p_emb, p_emb_up, p_emb_down], dim=-1)
            p[..., :4] = decode_delta_map(p[..., :4], self.anchor_vec.to(p))
            p[..., :4] *= self.stride

            return p.view(nB, -1, p.shape[-1])


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_dict, nID=0, test_emb=False):
        super(Darknet, self).__init__()
        if isinstance(cfg_dict, str):
            cfg_dict = parse_model_cfg(cfg_dict)
        self.module_defs = cfg_dict
        self.module_defs[0]['nID'] = nID
        self.img_size = [int(self.module_defs[0]['width']), int(self.module_defs[0]['height'])]
        self.emb_dim = int(self.module_defs[0]['embedding_dim'])
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.loss_names = ['loss', 'box', 'conf', 'id', 'nT']
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        self.test_emb = test_emb

        self.classifier = nn.Linear(self.emb_dim, nID) if nID > 0 else None

    def forward(self, x, targets=None, targets_len=None):
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        is_training = (targets is not None) and (not self.test_emb)
        # img_size = x.shape[-1]
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                if is_training:  # get loss
                    targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
                    x, *losses = module[0](x, self.img_size, targets, self.classifier)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                elif self.test_emb:
                    if targets is not None:
                        targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
                    x = module[0](x, self.img_size, targets, self.classifier, self.test_emb)
                else:  # get detections
                    x = module[0](x, self.img_size)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            self.losses['nT'] /= 3
            output = [o.squeeze() for o in output]
            return sum(output), torch.Tensor(list(self.losses.values())).cuda()
        elif self.test_emb:
            return torch.cat(output, 0)
        return torch.cat(output, 1)


def shift_tensor_vertically(t, delta):
    # t should be a 5-D tensor (nB, nA, nH, nW, nC)
    res = torch.zeros_like(t)
    if delta >= 0:
        res[:, :, :-delta, :, :] = t[:, :, delta:, :, :]
    else:
        res[:, :, -delta:, :, :] = t[:, :, :delta, :, :]
    return res


def create_grids(self, img_size, nGh, nGw):
    self.stride = img_size[0] / nGw
    assert self.stride == img_size[1] / nGh, \
        "{} v.s. {}/{}".format(self.stride, img_size[1], nGh)

    # build xy offsets
    grid_x = torch.arange(nGw).repeat((nGh, 1)).view((1, 1, nGh, nGw)).float()
    grid_y = torch.arange(nGh).repeat((nGw, 1)).transpose(0, 1).view((1, 1, nGh, nGw)).float()
    # grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4)

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
    if not os.path.isfile(weights):
        try:
            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
        except IOError:
            print(weights + ' not found')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Open the weights file
    fp = open(weights, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]  # number of images seen during training
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen  # number of images seen during training
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)

    fp.close()


def parse_model_cfg(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            if value[0] == '$':
                value = module_defs[0].get(value.strip('$'), None)
            module_defs[-1][key.rstrip()] = value

    return module_defs


def parse_data_cfg(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def build_targets_max(target, anchor_wh, nA, nC, nGh, nGw):
    """
    returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
    """
    nB = len(target)  # number of images in batch

    txy = torch.zeros(nB, nA, nGh, nGw, 2).cuda()  # batch size, anchors, grid size
    twh = torch.zeros(nB, nA, nGh, nGw, 2).cuda()
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0).cuda()
    tcls = torch.ByteTensor(nB, nA, nGh, nGw, nC).fill_(0).cuda()  # nC = number of classes
    tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1).cuda()
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long().cuda()
        t = t[:, [0, 2, 3, 4, 5]]
        nTb = len(t)  # number of targets
        if nTb == 0:
            continue

        # gxy, gwh = t[:, 1:3] * nG, t[:, 3:5] * nG
        gxy, gwh = t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh
        gi = torch.clamp(gxy[:, 0], min=0, max=nGw - 1).long()
        gj = torch.clamp(gxy[:, 1], min=0, max=nGh - 1).long()

        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        # gi, gj = torch.clamp(gxy.long(), min=0, max=nG - 1).t()
        # gi, gj = gxy.long().t()

        # iou of targets-anchors (using wh only)
        box1 = gwh
        box2 = anchor_wh.unsqueeze(1)
        inter_area = torch.min(box1, box2).prod(2)
        iou = inter_area / (box1.prod(1) + box2.prod(2) - inter_area + 1e-16)

        # Select best iou_pred and anchor
        iou_best, a = iou.max(0)  # best anchor [0-2] for each target

        # Select best unique target-anchor combinations
        if nTb > 1:
            _, iou_order = torch.sort(-iou_best)  # best to worst

            # Unique anchor selection
            u = torch.stack((gi, gj, a), 0)[:, iou_order]
            # _, first_unique = np.unique(u, axis=1, return_index=True)  # first unique indices
            first_unique = return_torch_unique_index(u, torch.unique(u, dim=1))  # torch alternative
            i = iou_order[first_unique]
            # best anchor must share significant commonality (iou) with target
            i = i[iou_best[i] > 0.60]  # TODO: examine arbitrary threshold
            if len(i) == 0:
                continue

            a, gj, gi, t = a[i], gj[i], gi[i], t[i]
            t_id = t_id[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        else:
            if iou_best < 0.60:
                continue

        tc, gxy, gwh = t[:, 0].long(), t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh

        # XY coordinates
        txy[b, a, gj, gi] = gxy - gxy.floor()

        # Width and height
        twh[b, a, gj, gi] = torch.log(gwh / anchor_wh[a])  # yolo method
        # twh[b, a, gj, gi] = torch.sqrt(gwh / anchor_wh[a]) / 2 # power method

        # One-hot encoding of label
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1
        tid[b, a, gj, gi] = t_id.unsqueeze(1)
    tbox = torch.cat([txy, twh], -1)
    return tconf, tbox, tid

def decode_delta(delta, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    return torch.stack([gx, gy, gw, gh], dim=1)

def encode_delta(gt_box_list, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                     gt_box_list[:, 2], gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw/pw)
    dh = torch.log(gh/ph)
    return torch.stack([dx, dy, dw, dh], dim=1)

def build_targets_thres(target, anchor_wh, nA, nC, nGh, nGw):
    ID_THRESH = 0.5
    FG_THRESH = 0.5
    BG_THRESH = 0.4
    nB = len(target)  # number of images in batch
    assert (len(anchor_wh) == nA)

    tbox = torch.zeros(nB, nA, nGh, nGw, 4).cuda()  # batch size, anchors, grid size
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0).cuda()
    tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1).cuda()
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long().cuda()
        t = t[:, [0, 2, 3, 4, 5]]
        nTb = len(t)  # number of targets
        if nTb == 0:
            continue

        gxy, gwh = t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh
        gxy[:, 0] = torch.clamp(gxy[:, 0], min=0, max=nGw - 1)
        gxy[:, 1] = torch.clamp(gxy[:, 1], min=0, max=nGh - 1)

        gt_boxes = torch.cat([gxy, gwh], dim=1)  # Shape Ngx4 (xc, yc, w, h)

        anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)
        anchor_list = anchor_mesh.permute(0, 2, 3, 1).contiguous().view(-1, 4)  # Shpae (nA x nGh x nGw) x 4
        # print(anchor_list.shape, gt_boxes.shape)
        iou_pdist = bbox_iou(anchor_list, gt_boxes)  # Shape (nA x nGh x nGw) x Ng
        iou_max, max_gt_index = torch.max(iou_pdist, dim=1)  # Shape (nA x nGh x nGw), both

        iou_map = iou_max.view(nA, nGh, nGw)
        gt_index_map = max_gt_index.view(nA, nGh, nGw)

        # nms_map = pooling_nms(iou_map, 3)

        id_index = iou_map > ID_THRESH
        fg_index = iou_map > FG_THRESH
        bg_index = iou_map < BG_THRESH
        ign_index = (iou_map < FG_THRESH) * (iou_map > BG_THRESH)
        tconf[b][fg_index] = 1
        tconf[b][bg_index] = 0
        tconf[b][ign_index] = -1

        gt_index = gt_index_map[fg_index]
        gt_box_list = gt_boxes[gt_index]
        gt_id_list = t_id[gt_index_map[id_index]]
        # print(gt_index.shape, gt_index_map[id_index].shape, gt_boxes.shape)
        if torch.sum(fg_index) > 0:
            tid[b][id_index] = gt_id_list.unsqueeze(1)
            fg_anchor_list = anchor_list.view(nA, nGh, nGw, 4)[fg_index]
            delta_target = encode_delta(gt_box_list, fg_anchor_list)
            tbox[b][fg_index] = delta_target
    return tconf, tbox, tid

def decode_delta_map(delta_map, anchors):
    '''
    :param: delta_map, shape (nB, nA, nGh, nGw, 4)
    :param: anchors, shape (nA,4)
    '''
    nB, nA, nGh, nGw, _ = delta_map.shape
    anchor_mesh = generate_anchor(nGh, nGw, anchors)
    anchor_mesh = anchor_mesh.permute(0,2,3,1).contiguous()              # Shpae (nA x nGh x nGw) x 4
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(nB,1,1,1,1)
    pred_list = decode_delta(delta_map.view(-1,4), anchor_mesh.view(-1,4))
    pred_map = pred_list.view(nB, nA, nGh, nGw, 4)
    return pred_map

def generate_anchor(nGh, nGw, anchor_wh):
    nA = len(anchor_wh)
    yy, xx =torch.meshgrid(torch.arange(nGh), torch.arange(nGw))
    xx, yy = xx.cuda(), yy.cuda()

    mesh = torch.stack([xx, yy], dim=0)                                              # Shape 2, nGh, nGw
    mesh = mesh.unsqueeze(0).repeat(nA,1,1,1).float()                                # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nGh,nGw) # Shape nA x 2 x nGh x nGw
    anchor_mesh = torch.cat([mesh, anchor_offset_mesh], dim=1)                       # Shape nA x 4 x nGh x nGw
    return anchor_mesh


def return_torch_unique_index(u, uv):
    n = uv.shape[1]  # number of columns
    first_unique = torch.zeros(n, device=u.device).long()
    for j in range(n):
        first_unique[j] = (uv[:, j:j + 1] == u).all(0).nonzero()[0]

    return first_unique

def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1,1).expand(N,M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1,-1).expand(N,M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height)/shape[0], float(width)/shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio)) # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if not os.path.isfile(path):
            raise FileExistsError

        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = self.get_size(self.vw, self.vh, self.width, self.height)
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
        img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of files


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, method='standard'):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    Args:
        prediction,
        conf_thres,
        nms_thres,
        method = 'standard' or 'fast'
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Non-maximum suppression
        nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        #
        # if method == 'standard':
        #     nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        # elif method == 'fast':
        #     nms_indices = fast_nms(pred[:, :4], pred[:, 4], iou_thres=nms_thres, conf_thres=conf_thres)
        # else:
        #     raise ValueError('Invalid NMS type!')
        det_max = pred[nms_indices]

        if len(det_max) > 0:
            # Add max detections to outputs
            output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    return output

def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords

def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    # x, y are coordinates of center
    # (x1, y1) and (x2, y2) are coordinates of bottom left and top right respectively.
    y = torch.zeros_like(x) if x.dtype is torch.float32 else np.zeros_like(x)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)  # Bottom left x
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)  # Bottom left y
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)  # Top right x
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)  # Top right y
    return y

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d num: %d' % (frame_id, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im