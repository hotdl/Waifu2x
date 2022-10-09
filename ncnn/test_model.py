import os

import cv2
import torch
import torchvision

from Models import UpConv_7


def img2tensor(img):
    # handle numpy array
    if img.ndim == 2:
        img = img[:, :, None]

    tensor = torch.from_numpy(img.transpose((2, 0, 1))).contiguous()
    # backward compatibility
    if isinstance(tensor, torch.ByteTensor):
        return tensor.to(dtype=torch.get_default_dtype()).div(255)
    else:
        return tensor


def tensor2img(tensor, **kwargs):
    grid = torchvision.utils.make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return ndarr


def test_model():
    print('test_model')
    model = UpConv_7()
    model.load_pre_train_weights("../model_check_points/Upconv_7/photo/noise2_scale2.0x_model.json")
    print(model)

    img = cv2.imread('../images/lena.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    input_tensor = img2tensor(img).unsqueeze(0)
    print(input_tensor.shape)

    output_tensor = model(input_tensor)
    print(output_tensor.shape)

    output_img = tensor2img(output_tensor)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    os.makedirs('../output', exist_ok=True)
    cv2.imwrite("../output/lena_out.png", output_img)


if __name__ == '__main__':
    test_model()
