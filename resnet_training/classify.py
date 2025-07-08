from PIL import Image
import argparse
import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.backends.cudnn as cudnn

from image_classification import models
import torchvision.transforms as transforms

from image_classification.models import (
    resnet50,
    resnext101_32x4d,
    se_resnext101_32x4d,
    efficientnet_b0,
    efficientnet_b4,
    efficientnet_widese_b0,
    efficientnet_widese_b4,
    efficientnet_quant_b0,
    efficientnet_quant_b4,
)


def available_models():
    models = {
        m.name: m
        for m in [
            resnet50,
            resnext101_32x4d,
            se_resnext101_32x4d,
            efficientnet_b0,
            efficientnet_b4,
            efficientnet_widese_b0,
            efficientnet_widese_b4,
            efficientnet_quant_b0,
            efficientnet_quant_b4,
        ]
    }
    return models


def add_parser_arguments(parser):
    model_names = available_models().keys()
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )
    parser.add_argument("--num_classes", default=3, type=int, help="Number of output classes")
    parser.add_argument("--precision", metavar="PREC", default="AMP", choices=["AMP", "FP32"])
    parser.add_argument("--cpu", action="store_true", help="perform inference on CPU")
    parser.add_argument("--image", metavar="<path>", help="path to classified image")


def load_jpeg_from_file(path, image_size, cuda=True):
    img_transforms = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    img = img_transforms(Image.open(path))
    with torch.no_grad():
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if cuda:
            mean = mean.cuda()
            std = std.cuda()
            img = img.cuda()
        img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)

    return input


def check_quant_weight_correctness(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = {
        k[len("module."):] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    quantizers_sd_keys = {
        f"{n[0]}._amax" for n in model.named_modules() if "quantizer" in n[0]
    }
    sd_all_keys = quantizers_sd_keys | set(model.state_dict().keys())
    assert set(state_dict.keys()) == sd_all_keys, (
        f"Passed quantized architecture, but following keys are missing in "
        f"checkpoint: {list(sd_all_keys - set(state_dict.keys()))}"
    )


def main(args, model_args):
    model_args.num_classes = args.num_classes

    try:
        model = available_models()[args.arch](**model_args.__dict__)
    except RuntimeError as e:
        print_in_box("Error when creating model, did you forget to run checkpoint2model script?")
        raise e

    if args.arch in ["efficientnet-quant-b0", "efficientnet-quant-b4"]:
        check_quant_weight_correctness(model_args.pretrained_from_file, model)

    if not args.cpu:
        model = model.cuda()
    model.eval()

    input = load_jpeg_from_file(args.image, args.image_size, cuda=not args.cpu)

    with torch.no_grad(), autocast(enabled=args.precision == "AMP"):
        output = torch.nn.functional.softmax(model(input), dim=1)

    output = output.float().cpu().view(-1).numpy()
    top_k = np.argsort(output)[::-1]

    print(f"\nPrediction for: {args.image}\n")
    for i in range(args.num_classes):
        idx = top_k[i]
        print(f"Class {idx}: {100 * output[idx]:.2f}%")

def print_in_box(msg):
    print("#" * (len(msg) + 10))
    print(f"#### {msg} ####")
    print("#" * (len(msg) + 10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Custom Classifier Inference")

    add_parser_arguments(parser)
    args, rest = parser.parse_known_args()
    model_args, rest = available_models()[args.arch].parser().parse_known_args(rest)

    assert len(rest) == 0, f"Unknown args passed: {rest}"

    cudnn.benchmark = True

    main(args, model_args)

