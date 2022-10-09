import os

import torch

from Models import UpConv_7


def export_onnx(weight_path, out_path):
    print('export onnx start')
    model = UpConv_7()
    model.load_pre_train_weights(weight_path)

    x = torch.randn((1, 3, 128, 128))
    torch.onnx.export(model, x, out_path,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=11)


def export_onnx_batch():
    styles = ['photo', 'anime']
    root_dir = '../model_check_points/Upconv_7'
    out_dir = 'onnx_model'

    for style in styles:
        if not os.path.exists(os.path.join(out_dir, style)):
            os.makedirs(os.path.join(out_dir, style), exist_ok=True)
        for i in range(4):
            weight_path = f'{root_dir}/{style}/noise{i}_scale2.0x_model.json'
            out_path = f'{out_dir}/{style}/noise{i}.onnx'
            print(weight_path, out_path)
            export_onnx(weight_path, out_path)
        export_onnx(f'{root_dir}/{style}/scale2.0x_model.json', f'{out_dir}/{style}/scale.onnx')


if __name__ == '__main__':
    export_onnx_batch()
