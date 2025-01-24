from utils import convert_onnx_to_engine

onnx_file_path ="/root/autodl-tmp/models/tensorrt/onnx/BiRefNet-matting-epoch_100.onnx"
engine_file_path = "/root/autodl-tmp/models/tensorrt/BiRefNet/BiRefNet-Matting_trt.trt"

convert_onnx_to_engine(onnx_file_path, engine_file_path)

#python /root/autodl-tmp/models/tensorrt/birefnet_tensorrt/engine_converter.py