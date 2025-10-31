import torch
import numpy as np
import os
import torch.distributed
import torchvision
from torchvision.transforms import functional as t_F
import torch.nn.functional as F
import random
import math

# keep top k largest values, and smooth others
def keep_top_k(p, k, n_classes=1000):  # p is the softmax on label output
    if k == n_classes:
        return p

    values, indices = p.topk(k, dim=1)

    mask_topk = torch.zeros_like(p)
    mask_topk.scatter_(-1, indices, 1.0)
    top_p = mask_topk * p

    minor_value = (1 - torch.sum(values, dim=1)) / (n_classes - k)
    minor_value = minor_value.unsqueeze(1).expand(p.shape)
    mask_smooth = torch.ones_like(p)
    mask_smooth.scatter_(-1, indices, 0)
    smooth_p = mask_smooth * minor_value

    topk_smooth_p = top_p + smooth_p
    assert np.isclose(
        topk_smooth_p.sum().item(), p.shape[0]
    ), f"{topk_smooth_p.sum().item()} not close to {p.shape[0]}"
    return topk_smooth_p


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,),type="train"):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(target,pred)
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    
    # if type=="val":
    #     print("pred:")
    #     print(pred)
    #     print("target")
    #     print(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_class(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    class_num = output.shape[1]

    class_total=torch.histc(target,bins=1000,min=0,max=999)
    tmp=correct[:1].reshape(-1).float()
    class_correct=target[torch.nonzero(tmp).squeeze()]
    class_correct=torch.histc(class_correct,bins=1000,min=0,max=999)
    return class_correct, class_total

def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find("weight") >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(
        group_no_weight_decay
    )
    groups = [
        dict(params=group_weight_decay),
        dict(params=group_no_weight_decay, weight_decay=0.0),
    ]
    return groups

class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, classes, ipc, mem=False, shuffle=False, **kwargs):
        super(ImageFolder, self).__init__(**kwargs)
        self.mem = mem
        self.image_paths = []
        self.targets = []
        self.samples = []
        self.iaug = 16 # 12 directions
        self.augsamples = []
        classidx = os.listdir(self.root)
        if self.root=="/root/autodl-tmp/RDED-main/dataset/imagenet/val":
            classidx = ["n02096294","n02093754","n02111889","n02088364","n02086240","n02089973","n02087394","n02115641","n02099601","n02105641"] ## imagewoof, synthesize时去掉注释
            # classidx = ["n01440764","n02102040","n02979186","n03000684","n03028079","n03394916","n03417042","n03425413","n03445777","n03888257"] ## imagenette
            # classidx = ['n02869837', 'n01749939', 'n02488291', 'n02107142', 'n13037406', 'n02091831', 'n04517823', 'n04589890', 'n03062245', 'n01773797'] # imagenet-10
            # classidx = ["n02123045","n02123159","n02123394","n02123597","n02124075","n02129165","n02129604","n02128925","n02128757","n02127052"] ## imagecat
        elif self.root=="/root/autodl-tmp/RDED-main/dataset/imagenet/train":
            classidx = ["n02096294","n02093754","n02111889","n02088364","n02086240","n02089973","n02087394","n02115641","n02099601","n02105641"] ## imagewoof, synthesize时去掉注释
            # classidx = ["n01440764","n02102040","n02979186","n03000684","n03028079","n03394916","n03417042","n03425413","n03445777","n03888257"] ## imagenette
            # classidx = ['n02869837', 'n01749939', 'n02488291', 'n02107142', 'n13037406', 'n02091831', 'n04517823', 'n04589890', 'n03062245', 'n01773797'] # imagenet-10
            # classidx = ["n02123045","n02123159","n02123394","n02123597","n02124075","n02129165","n02129604","n02128925","n02128757","n02127052"] ## imagecat
        else:
            classidx.sort()
        
        for c in range(len(classes)):
            # dir_path = self.root + "/" + str(classes[c]).zfill(5)
            
            dir_path = self.root + "/" + str(classidx[c])
            # print(dir_path)
            file_ls = os.listdir(dir_path)
            if shuffle:
                random.shuffle(file_ls)
            # print(len(file_ls))
            for i in range(ipc):
                if file_ls[i]==".ipynb_checkpoints":
                    continue
                self.image_paths.append(dir_path + "/" + file_ls[i])
                self.targets.append(c)
                if self.mem:
                    self.samples.append(self.loader(dir_path + "/" + file_ls[i]))
                        
    def __getitem__(self, index):
        if self.mem:
            sample = self.samples[index]
        else:
            sample = self.loader(self.image_paths[index])
        if self.root=='./exp/imagenet-woof_resnet18_f2_mipc300_ipc50_cr5/syn_data':
            tot = 224
            hal = tot//2
            h1 = hal//2
            h2 = (hal+tot)//2
            
            sample_origin = sample.copy()
            aug_sample = sample.copy() # down
            aug_sample_1 = sample.copy() # up
            aug_sample_2 = sample.copy() # left
            aug_sample_3 = sample.copy() # right 
            aug_sample_4 = sample.copy() # up left
            aug_sample_5 = sample.copy() # down left
            aug_sample_6 = sample.copy() # up right
            aug_sample_7 = sample.copy() # down right
            aug_sample_8 = sample.copy() # up up
            aug_sample_9 = sample.copy() # up down
            aug_sample_10 = sample.copy() # down up
            aug_sample_11 = sample.copy() # down down
            aug_sample_12 = sample.copy() # 1
            aug_sample_13 = sample.copy() # 2
            aug_sample_14 = sample.copy() # 3
            aug_sample_15 = sample.copy() # 4


            down=sample.crop((0,hal,tot,tot))
            up=sample.crop((0,0,tot,hal))
            left=sample.crop((0,0,hal,tot))
            right=sample.crop((hal,0,tot,tot))
            ul = sample.crop((0,0,hal,hal))
            dl = sample.crop((0,hal,hal,tot))
            ur = sample.crop((hal,0,tot,hal))
            dr = sample.crop((hal,hal,tot,tot))
            uu = sample.crop((0,0,tot,h1))
            ud = sample.crop((0,h1,tot,hal))
            du = sample.crop((0,hal,tot,h2))
            dd = sample.crop((0,h2,tot,tot))

            aug_sample.paste(down,(0,0))
            aug_sample_1.paste(up,(0,hal))
            aug_sample_2.paste(left,(hal,0))
            aug_sample_3.paste(right,(0,0))
            aug_sample_4.paste(ul,(hal,0)),aug_sample_4.paste(ul,(0,hal)),aug_sample_4.paste(ul,(hal,hal))
            aug_sample_5.paste(dl,(0,0)),aug_sample_5.paste(dl,(hal,0)),aug_sample_5.paste(dl,(hal,hal))
            aug_sample_6.paste(ur,(hal,hal)),aug_sample_6.paste(ur,(0,hal)),aug_sample_6.paste(ur,(0,0))
            aug_sample_7.paste(dr,(0,hal)),aug_sample_7.paste(dr,(0,0)),aug_sample_7.paste(dr,(hal,0))

            aug_sample_8.paste(uu,(0,h1)),aug_sample_8.paste(uu,(0,hal)),aug_sample_8.paste(uu,(0,h2))
            aug_sample_9.paste(ud,(0,0)),aug_sample_9.paste(ud,(0,hal)),aug_sample_9.paste(ud,(0,h2))
            aug_sample_10.paste(du,(0,0)),aug_sample_10.paste(du,(0,h1)),aug_sample_10.paste(du,(0,h2))
            aug_sample_11.paste(dd,(0,0)),aug_sample_11.paste(dd,(0,h1)),aug_sample_11.paste(dd,(0,hal))
            
            aug_sample_12 = ul.resize((tot,tot))
            aug_sample_13 = dl.resize((tot,tot))
            aug_sample_14 = ur.resize((tot,tot))
            aug_sample_15 = dr.resize((tot,tot))
            
            sample_origin = self.transform(sample_origin)
            aug_sample = self.transform(aug_sample)
            aug_sample_1 = self.transform(aug_sample_1)
            aug_sample_2 = self.transform(aug_sample_2)
            aug_sample_3 = self.transform(aug_sample_3)
            aug_sample_4 = self.transform(aug_sample_4)
            aug_sample_5 = self.transform(aug_sample_5)
            aug_sample_6 = self.transform(aug_sample_6)
            aug_sample_7 = self.transform(aug_sample_7)
            aug_sample_8 = self.transform(aug_sample_8)
            aug_sample_9 = self.transform(aug_sample_9)
            aug_sample_10 = self.transform(aug_sample_10)
            aug_sample_11 = self.transform(aug_sample_11)
            aug_sample_12 = self.transform(aug_sample_12)
            aug_sample_13 = self.transform(aug_sample_13)
            aug_sample_14 = self.transform(aug_sample_14)
            aug_sample_15 = self.transform(aug_sample_15)
            
            # sample = torch.cat([sample_origin,aug_sample,aug_sample_1,aug_sample_2,aug_sample_3,aug_sample_4,aug_sample_5,aug_sample_6,aug_sample_7,aug_sample_8,aug_sample_9,aug_sample_10,aug_sample_11,aug_sample_12,aug_sample_13,aug_sample_14,aug_sample_15])
            sample = torch.cat([sample_origin,sample_origin])
            # copy up
            # sample_copy = sample.copy()
            # aug_sample = sample.copy()
            # sample_copy = self.transform(sample_copy)
            # up=sample.crop((0,0,224,112))
            # aug_sample.paste(up,(0,112))
            # aug_sample = self.transform(aug_sample)
            # sample = torch.cat([sample_copy,aug_sample])
        else:
            sample = self.transform(sample)
        return sample, self.targets[index]

    def __len__(self):
        return len(self.targets)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(images, args, rand_index=None, lam=None, bbox=None):
    rand_index = torch.randperm(images.size()[0]).cuda()
    lam = np.random.beta(args.cutmix, args.cutmix)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]


def mixup(images, args, rand_index=None, lam=None):
    rand_index = torch.randperm(images.size()[0]).cuda()
    lam = np.random.beta(args.mixup, args.mixup)

    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, rand_index.cpu(), lam, None


def mix_aug(images, args, rand_index=None, lam=None, bbox=None):
    if args.mix_type == "mixup":
        return mixup(images, args, rand_index, lam)
    elif args.mix_type == "cutmix":
        return cutmix(images, args, rand_index, lam, bbox)
    else:
        return images, None, None, None


class ShufflePatches(torch.nn.Module):
    def shuffle_weight(self, img, factor):
        h, w = img.shape[1:]
        th, tw = h // factor, w // factor
        patches = []
        for i in range(factor):
            i = i * tw
            if i != factor - 1:
                patches.append(img[..., i : i + tw])
            else:
                patches.append(img[..., i:])
        random.shuffle(patches)
        img = torch.cat(patches, -1)
        return img

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        img = self.shuffle_weight(img, self.factor)
        img = img.permute(0, 2, 1)
        return img
