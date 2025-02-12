import json
import os

# https://huggingface.co/EmmaJohnson311/TensorRT-ONNX-collect/tree/main

class ONNX_ModelData():
    def __init__(self,json_path = None):
        self.json_path = json_path
        if self.json_path is None:
            self.json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'onnx_data.json')
        self.data = None
        self.model_class = None
        self.onnx_data = None

        self.model_collect = None #带分类结构的字典
        self.SHA256_dict = None #模型名对应哈希的字典

        self._load_json()
        self._get_collect_list()
        self._get_SHA256_dict()

    def _load_json(self):
        with open(self.json_path, 'r') as file:
            self.data = json.load(file)
        self.model_class = self.data["trt_path"]
        self.onnx_data = self.data["onnx_data"]
        
    def _get_collect_list(self):
        self.model_collect = {}
        for i in self.model_class.keys():
            self.model_collect[i] = list(self.onnx_data[i].keys())

    def _get_SHA256_dict(self):
        self.SHA256_dict = {}
        for k,v in self.onnx_data.items():
            for i,j in v.items():
                self.SHA256_dict[i] = j["SHA256"]

    def contrast_SHA256(self,input_file,SHA256): #输入下载的文件，对比原哈希值是否相同
        ...

    def lookup_SHA256(self,SHA256_or_file): #通过SHA256查询模型名
        ...

    def Download_Progress(self,file_path): #通过输入的文件路径作为存放路径在hf下载文件，且通过完整文件大小来实时显示进度
        ...
    

test = ONNX_ModelData()
SHA256_dict = test.SHA256_dict
model_class = test.model_class
model_collect = test.model_collect


def BiRefNet_name():
    pretrained_weights = [
    'zhengpeng7/BiRefNet',
    'zhengpeng7/BiRefNet-portrait',
    'zhengpeng7/BiRefNet-legacy', 
    'zhengpeng7/BiRefNet-DIS5K-TR_TEs', 
    'zhengpeng7/BiRefNet-DIS5K',
    'zhengpeng7/BiRefNet-HRSOD',
    'zhengpeng7/BiRefNet-COD',
    'zhengpeng7/BiRefNet_lite',     # Modify the `bb` in `config.py` to `swin_v1_tiny`.
    ]
    # https://objects.githubusercontent.com/github-production-release-asset-2e65be/525717745/81693dcf-8d42-4ef6-8dba-1f18f87de174?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20241014%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241014T003944Z&X-Amz-Expires=300&X-Amz-Signature=ec867061341cf6498cf5740c36f49da22d4d3d541da48d6e82c7bce0f3b63eaf&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3DBiRefNet-COD-epoch_125.pth&response-content-type=application%2Foctet-stream

    return pretrained_weights