import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from graphs.models.deeplab_multi import DeeplabMulti
from graphs.models.deeplab101_IN import DeeplabMulti101_IN
from graphs.models.deeplab50_IN import DeeplabMulti50_IN
from graphs.models.deeplab50_bn import Deeplab50_bn
from graphs.models.sw_101 import Deeplab101_sw
from graphs.models.auto_multi import Auto_res101

def get_model(args):
    if args.backbone == "deeplabv2_multi":
        model = DeeplabMulti(args,num_classes=args.num_classes,
                            pretrained=args.imagenet_pretrained)
        params = model.optim_parameters(args)
        args.numpy_transform = True
    elif args.backbone == "Deeplab101_IN":
        model = DeeplabMulti101_IN(args,num_classes=args.num_classes,
                            pretrained=args.imagenet_pretrained)
        params = model.optim_parameters(args)
        args.numpy_transform = True
    elif args.backbone == "Deeplab50_IN":
        model = DeeplabMulti50_IN(args,num_classes=args.num_classes,
                            pretrained=args.imagenet_pretrained)
        params = model.optim_parameters(args)
        args.numpy_transform = True
    elif args.backbone == "Deeplab50_bn":
        model = Deeplab50_bn(args,num_classes=args.num_classes,
                            pretrained=args.imagenet_pretrained)
        params = model.optim_parameters(args)
        args.numpy_transform = True

    if args.backbone == "SW_101":
        model = Deeplab101_sw(args,num_classes=args.num_classes,
                            pretrained=args.imagenet_pretrained)
        params = model.optim_parameters(args)
        args.numpy_transform = True

    if args.backbone == "auto_101":
        model =Auto_res101(args,num_classes=args.num_classes,
                            pretrained=args.imagenet_pretrained)
        params = model.optim_parameters(args)
        args.numpy_transform = True
    return model, params