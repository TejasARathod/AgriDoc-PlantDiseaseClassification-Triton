import argparse
import torch
import pytorch_quantization

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


def parse_args(parser):
    model_names = available_models().keys()
    parser.add_argument("--arch", "-a", metavar="ARCH", default="resnet50", choices=model_names,
                        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                        help="device to use for model export")
    parser.add_argument("--image-size", default=224, type=int,
                        help="Input image resolution (default: 224)")
    parser.add_argument("--output", type=str, help="Path to save exported ONNX model")
    parser.add_argument("-b", "--batch-size", default=1, type=int,
                        help="Dummy batch size for export shape (default: 1)")
    parser.add_argument("--num-classes", type=int, required=True, help="number of output classes")
    return parser


def final_name(base_name):
    if base_name is None:
        return "exported_model.onnx"
    parts = base_name.split(".")
    if "pt" in parts:
        return base_name.replace("pt", "onnx")
    elif "pth" in parts:
        return base_name.replace("pth", "onnx")
    elif len(parts) > 1:
        return ".".join(parts[:-1] + ["onnx"])
    else:
        return base_name + ".onnx"


def check_quant_weight_correctness(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    quantizers_sd_keys = {f"{n[0]}._amax" for n in model.named_modules() if "quantizer" in n[0]}
    sd_all_keys = quantizers_sd_keys | set(model.state_dict().keys())
    assert set(state_dict.keys()) == sd_all_keys, (
        f"Quantized model is missing keys: {list(sd_all_keys - set(state_dict.keys()))}"
    )


def main(args, model_args, model_arch):
    quant_arch = args.arch in ["efficientnet-quant-b0", "efficientnet-quant-b4"]
    if quant_arch:
        pytorch_quantization.nn.modules.tensor_quantizer.TensorQuantizer.use_fb_fake_quant = True

    # IMPORTANT: Override model_args.num_classes with args.num_classes
    model_args.num_classes = args.num_classes

    model = model_arch(**model_args.__dict__)
    model.to(args.device)
    model.eval()

    if quant_arch and model_args.pretrained_from_file is not None:
        check_quant_weight_correctness(model_args.pretrained_from_file, model)

    image_size = args.image_size
    dummy_input = torch.randn(args.batch_size, 3, image_size, image_size).to(args.device)

    final_model_path = args.output if args.output else final_name(model_args.pretrained_from_file)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            final_model_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            opset_version=13,
            verbose=True,
            do_constant_folding=True
        )
    print(f"Model exported to {final_model_path}")


if __name__ == "__main__":
    epilog = [
        "Based on the architecture picked by --arch flag, you may use the following options:\n"
    ]
    for model, ep in available_models().items():
        model_help = "\n".join(ep.parser().format_help().split("\n")[2:])
        epilog.append(model_help)

    parser = argparse.ArgumentParser(
        description="Export PyTorch model to ONNX with dynamic batch support",
        epilog="\n".join(epilog),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser = parse_args(parser)
    args, rest = parser.parse_known_args()

    model_arch = available_models()[args.arch]
    model_args, rest = model_arch.parser().parse_known_args(rest)

    assert len(rest) == 0, f"Unknown args passed: {rest}"
    main(args, model_args, model_arch)

