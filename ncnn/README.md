# Waifu2x的pytorch模型转ncnn



## python依赖

numpy==1.19.5

opencv-contrib-python==4.5.5.64

opencv-python==4.5.5.64

torch==1.8.1

torchaudio==0.8.1

torchvision==0.9.1



## pytorch转onnx

直接执行export_onnx.py 即可，模型生成在onnx_model文件夹



## onnx精简

使用在线工具https://www.convertmodel.com/

进行onnx模型精简



## onnx转ncnn

根据https://github.com/Tencent/ncnn/wiki/how-to-build

安装ncnn，设置bin目录到系统路径

执行export_ncnn.sh，模型生成在ncnn_model文件夹

