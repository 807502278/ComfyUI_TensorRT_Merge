import os
from utils import convert_onnx_to_engine

#onnx_file_path ="/root/autodl-fs/models/tensorrt/BiRefNet_py/BiRefNet-main/BiRefNet-matting-epoch_100_pix384.onnx"
#engine_file_path = "/root/autodl-fs/models/tensorrt/BiRefNet/BiRefNet-matting-epoch_100_pix384.engine.trt"
onnx_list = ["BiRefNet-matting-epoch_100_pix384",
             "BiRefNet-matting-epoch_100_pix448",
             "BiRefNet-matting-epoch_100_pix480",
             "BiRefNet-matting-epoch_100_pix512",
             "BiRefNet-matting-epoch_100_pix768"]

for i in range(len(onnx_list)):
    onnx_path = os.path.join("/root/autodl-tmp/BiRefNet_py/BiRefNet-main",onnx_list[i] + ".onnx")
    engine_path =  os.path.join("/root/autodl-tmp/BiRefNet_model",onnx_list[i] + ".engine.trt")
    print(f"**********************\n开始构建第{i+1}-{len(onnx_list)}个：\n{onnx_path}\n{engine_path}")
    try:
        convert_onnx_to_engine(onnx_path, engine_path)
        print(f"第{i+1}-{len(onnx_list)}个构建成功！")
    except:
        print(f"第{i+1}-{len(onnx_list)}个构建失败！")