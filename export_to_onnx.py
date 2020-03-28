import argparse
import io
import torch
from torch.autograd import Variable
import onnx

from ssd import build_ssd


def assertONNXExpected(binary_pb):
    model_def = onnx.ModelProto.FromString(binary_pb)
    onnx.helper.strip_doc_string(model_def)
    return model_def


def export_to_string(model, inputs, version=None):
    f = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, inputs, f, export_params=True, opset_version=version)
    return f.getvalue()


def save_model(model, input, output):
    onnx_model_pb = export_to_string(model, input)
    model_def = assertONNXExpected(onnx_model_pb)
    with open(output, 'wb') as file:
        file.write(model_def.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Export trained model to ONNX format')
    parser.add_argument('--model', required=True, help='Path to saved PyTorch network weights (*.pth)')
    parser.add_argument('--output', default='ssd.onnx', help='Name of ouput file')
    parser.add_argument('--size', default=300, help='Input resolution')
    parser.add_argument('--num_classes', default=21, help='Number of trained classes + 1 for background')
    args = parser.parse_args()

    net = build_ssd('export', args.size, args.num_classes)
    net.load_state_dict(torch.load(args.model, map_location='cpu'))
    net.eval()

    input = Variable(torch.randn(1, 3, args.size, args.size))
    save_model(net, input, args.output)
