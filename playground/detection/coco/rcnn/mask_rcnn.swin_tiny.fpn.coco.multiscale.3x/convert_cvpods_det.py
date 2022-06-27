import torch
import argparse

parser = argparse.ArgumentParser("Tool to convert the backbone model")
parser.add_argument(
    "--path",
    help="path to the pytorch checkpoint file",
    type=str,
)
args, _ = parser.parse_known_args()
file_path = args.path
ckpt = torch.load(file_path, map_location='cpu')
new_ckpt = dict()
for key, value in ckpt['model'].items():
    new_ckpt['backbone.bottom_up.'+key] = value

output_path = file_path.replace(".pth", "_cvpods_det.pth")
torch.save({'model': new_ckpt}, output_path)
