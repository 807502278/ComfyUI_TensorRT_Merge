import os
import time

from typing import List
import numpy as np
from PIL import Image
import torch

import folder_paths
from .model_data.load_data import model_collect,SHA256_dict,model_class
from .export_trt.utilities import Engine
#from .model_data.util import tensor2pil,tensor2np
#from .BiRefNet.models.birefnet import BiRefNet


torch.set_float32_matmul_precision(["high", "highest"][0])
CATEGORY_NAME = "TensoRT/building"


class Building_TRT:
    DESCRIPTION = """
    onnx模型转trt节点说明：
        将抱脸TensorRT-ONNX-collect项目内的onnx模型按原目录结构放在:
            /autodl-fs/data/models/tensorrt/TensorRT-ONNX-collect
        将自动使用不同的转换程序转换选择的模型select_model，并将加速模型放入默认加载路径
        若没有onnx模型，将自动从抱脸下载，MirrorDownload控制是否使用hf-mirror.com镜像下载
        默认force_building=False若有加速模型跳过转换，为True时不跳过转换而是每次都生成增加时间后缀的加速模型
    """
    @classmethod
    def INPUT_TYPES(cls):
        all_model_name = []
        for i in model_collect.values():
            all_model_name = all_model_name + i

        # #区分本地模型，待完成
        #all_model = glob.glob(os.path.join(folder_paths.models_dir,"tensorrt","*.onnx"))
        #all_model_name = []
        #for i in all_model:
        #    file_name = os.path.basename(i)
        #    files = os.path.dirname(i).split(os.sep)[-1]
        #    all_model_name.append(os.path.join(files,file_name))

        return {
            "required": {
                "select_model": (all_model_name,{"default": all_model_name[0],}),
                "force_building": ("BOOLEAN",{"default": False}),
                "use_fp16": ("BOOLEAN",{"default": True}),
                "MirrorDownload": ("BOOLEAN",{"default": True}),
            },
            "optional": {
                "AdvancedSetting": ("TRT_SET",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("trt_path",)
    OUTPUT_NODE = True
    FUNCTION = "building"
    CATEGORY = CATEGORY_NAME
    def building(self,select_model,force_building,use_fp16,MirrorDownload,AdvancedSetting=None):
        download_path_onnx = os.path.join(folder_paths.models_dir,"tensorrt/TensorRT-ONNX-collect")
        # 准备路径和文件字符串
        # select_model_class = model_class[select_model.split("/")[0]]
        onnx_name = os.path.basename(select_model)
        trt_name = os.path.splitext(onnx_name)[0] + ".engine"
        select_model_class = model_class[select_model.split("/")[0]]
        new_path = self.mk_path(os.path.join(folder_paths.models_dir,"tensorrt",select_model_class))
        trt_path = os.path.join(new_path,trt_name)
        onnx_path = self.Download(download_path_onnx,select_model,MirrorDownload)

        if not force_building:
            if not os.path.isfile(trt_path): #没有trt则转换
                print(f"Prompt: Start Conversion {select_model}")
                self.Conversion(select_model_class,onnx_path,trt_path,use_fp16)
            print("Prompt: Detected existing trt model and force_building=False, skip conversion")
        else : #强制转换
            formatted_time = str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
            trt_name = os.path.splitext(onnx_name)[0] + formatted_time + ".engine"
            trt_path = os.path.join(new_path,trt_name)
            self.Conversion(select_model_class,onnx_path,trt_path,use_fp16)

        return (trt_path,)


    def Conversion(self,select_model_class,onnx_path,trt_path,use_fp16,Setting=None): #转换
        model = None
        rife_class = ["rife-onnx",model_class["rife-onnx"]]
        BiRefNet_class = ["BiRefNet-v2-onnx",model_class["BiRefNet-v2-onnx"]]
        scale_class = ["Upscaler-Onnx",model_class["Upscaler-Onnx"]]
        all_class = list(model_class.keys()) + list(model_class.values())
        if select_model_class in rife_class :  #rife面部重建
            print("Prompt: Current conversion model: Rife")
            model = self.building_4(onnx_path, trt_path, use_fp16)
        elif select_model_class in BiRefNet_class : #bf抠图
            print("Prompt: Current conversion model: BiRefNet v2")
            model = self.building_RGB(onnx_path, trt_path)
        elif select_model_class in scale_class : #缩放
            print(f"Prompt: Current conversion model: {select_model_class}")
            model = self.building_UpScale(onnx_path, trt_path, use_fp16)
        elif select_model_class in all_class :  #其它
            print(f"Prompt: Current conversion model: {select_model_class}")
            model = self.building_A(onnx_path, trt_path, use_fp16)
        else: #不支持的类型
            raise TypeError("Error: Unknown or unsupported type.")
        print(f"Prompt: Conversion successful! The model is saved in:\n {trt_path}")
        return model
    
    def mk_path(self,path): #创建路径
        if not os.path.isdir(path): os.mkdir(path)
        return path

    def Download(self,download_path_onnx,select_model,MirrorDownload=True):
        onnx_path = os.path.join(download_path_onnx,select_model)
        if not os.path.isfile(onnx_path): #若本地没有onnx模型则下载
            if MirrorDownload: 
                os.system("export HF_ENDPOINT='https://hf-mirror.com'")
                print(f"Download link:https://hf-mirror.com/EmmaJohnson311/TensorRT-ONNX-collect/{select_model}")
            else:
                print(f"Download link:https://huggingface.co/EmmaJohnson311/TensorRT-ONNX-collect/{select_model}")
            print(f"to:{onnx_path}")
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id= "EmmaJohnson311/TensorRT-ONNX-collect",
                            filename = select_model,
                            local_dir = download_path_onnx,
                            local_files_only=False)
            print("Download completed.")
        return onnx_path
        
    def building_A(self, onnx_path, trt_path, use_fp16):
        engine = Engine(trt_path)
        s = time.time()
        ret = engine.build(
            onnx_path,
            use_fp16,
            enable_preview=True,
        )
        e = time.time()
        print(f"Time taken to build: {(e-s)} seconds")
        print(f"Tensorrt engine saved at: {trt_path}")
        return ret

    def building_RGB(self, onnx_path, trt_path, Setting=None):
        import tensorrt as trt
        from tensorrt_bindings import Logger

        logger = Logger(Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, logger)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

        logger.log(trt.Logger.Severity.INFO, "Parse ONNX file")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.log(trt.Logger.ERROR, "ERROR: Failed to parse onnx file")
                for err in range(parser.num_errors):
                    logger.log(trt.Logger.ERROR, parser.get_error(err))
                raise RuntimeError("parse onnx file error")

        logger.log(trt.Logger.Severity.INFO, "Building TensorRT engine. This may take a few minutes.")
        engine_bytes = builder.build_serialized_network(network, config)

        trt_path = os.path.splitext(trt_path)[0] + ".trt"
        with open(trt_path, 'wb') as f:
            f.write(engine_bytes)
        return None

    def building_4(self, onnx_path, trt_path, use_fp16, Setting=None):
        engine = Engine(trt_path)
        torch.cuda.empty_cache()
        s = time.time()
        ret = engine.build(
            onnx_path,
            use_fp16,
            enable_preview=True,
            input_profile=[
                # any sizes from 256x256 to 3840x3840, batch size 1
                {
                    "img0": [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 3840, 3840)],
                    "img1": [(1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 3840, 3840)],
                },
            ],
        )
        e = time.time()
        print(f"Time taken to build: {(e-s)} seconds")
        print(f"Tensorrt engine saved at: {trt_path}")
        return ret
    
    def building_4_depth(self, onnx_path, trt_path, use_fp16):
        from .export_trt.trt_utilities import Engine_4
        engine = Engine_4(trt_path)

        s = time.time()
        ret = engine.build(
            onnx_path,
            use_fp16,
            enable_preview=True,
        )
        e = time.time()
        print(f"Time taken to build: {(e-s)} seconds")
        print(f"Tensorrt engine saved at: {trt_path}")
        return ret

    def building_UpScale(self, onnx_path, trt_path, use_fp16, Setting=None):
        from .Upscaler_.utilities import Engine_SC
        engine = Engine_SC(trt_path)
        torch.cuda.empty_cache()
        s = time.time()
        ret = engine.build(
            onnx_path,
            use_fp16,
            enable_preview=True,
            input_profile=[
                {"input": [(1,3,256,256), (1,3,512,512), (1,3,1280,1280)]}, # any sizes from 256x256 to 1280x1280
            ],
        )
        e = time.time()
        print(f"Time taken to build: {(e-s)} seconds")
        print(f"Tensorrt engine saved at: {trt_path}")
        return ret


class Custom_Building_TRT(Building_TRT):
    DESCRIPTION = """
    自定义onnx模型转trt说明：
        将自定义onnx模型放在:/autodl-fs/data/models/tensorrt/custom_onnx 可使用此节点加载
        另外需要选择类型：select_class 将此类型使用不同的转换程序转换，并将加速模型放入默认加载路径
    """
    custom_onnx_path = os.path.join(folder_paths.models_dir,"tensorrt","custom_onnx")
    @classmethod
    def INPUT_TYPES(cls):
        onnx_path = os.listdir(cls.mk_path(cls,cls.custom_onnx_path))
        if len(onnx_path)==0:
            onnx_path = ["* models/tensorrt/custom_onnx There are no files inside! *"]
        model_class_list = list(model_class.keys())
        return {
            "required": {
                "select_model": (onnx_path,{"default": onnx_path[0],}),
                "select_class":(model_class_list,{"default": model_class_list[0],}),
                "force_building": ("BOOLEAN",{"default": False}),
                "use_fp16": ("BOOLEAN",{"default": True}),
            },
            "optional": {
                "AdvancedSetting": ("TRT_SET",),
            },
        }
    
    def building(self,select_model,select_class,force_building,use_fp16,AdvancedSetting=None):
        # 准备路径和文件字符串
        onnx_name = os.path.basename(select_model)
        trt_name = os.path.splitext(onnx_name)[0] + ".engine"
        trt_name2 = os.path.splitext(onnx_name)[0] + ".trt"
        new_path = self.mk_path(os.path.join(folder_paths.models_dir,"tensorrt",model_class[select_class]))
        if not os.path.isdir(new_path): os.mkdir(new_path) #创建输出路径
        trt_path = os.path.join(new_path,trt_name)
        trt_path2 = os.path.join(new_path,trt_name2)
        onnx_path = os.path.join(self.__class__.custom_onnx_path,select_model)

        model = None
        if not force_building:
            if os.path.isfile(trt_path): #有.engine文件
                print("Prompt: Detected existing engine model and force_building=False, skip conversion")
            elif os.path.isfile(trt_path2): #有.trt文件
                print("Prompt: Detected existing trt model and force_building=False, skip conversion")
                trt_path = trt_path2
            else:
                print(f"Prompt: Start Conversion {select_model}")
                model = self.Conversion(select_class,onnx_path,trt_path,use_fp16)
        else : #强制转换
            formatted_time = str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
            trt_name = os.path.splitext(onnx_name)[0] + formatted_time + ".engine"
            trt_path = os.path.join(new_path,trt_name)
            model = self.Conversion(select_model,onnx_path,trt_path,use_fp16)
        return (model,)


#class General_TensorRT_Run: #待开发
#    ...
#
#class TRTset_Rife(): #待开发
#    ...
#
#class TRTset_BiRefNet(): #待开发
#    ...
#
#class TRTset_Upscaler(): #待开发
#    ...


NODE_CLASS_MAPPINGS = {
    "Building_TRT":Building_TRT,
    "Custom_Building_TRT":Custom_Building_TRT
}
