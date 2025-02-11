import os
import time

from typing import List
import numpy as np
from PIL import Image
import torch

import folder_paths
from .models_BiRefNet.models.birefnet import BiRefNet
from .export_trt.trt_utilities import Engine_4
from .export_trt.utilities import Engine

torch.set_float32_matmul_precision(["high", "highest"][0])

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
# Tensor to np
def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
  if len(tensor.shape) == 3:  # Single image
    return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
  else:  # Batch of images
    return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]

# https://huggingface.co/EmmaJohnson311/TensorRT-ONNX-collect/tree/main
model_collect = {
    "BiRefNet-v2-onnx" : ["BiRefNet-v2-onnx/BiRefNet_lite-general-2K-epoch_232.onnx",
                  "BiRefNet-v2-onnx/BiRefNet-COD-epoch_125.onnx",
                  "BiRefNet-v2-onnx/BiRefNet-DIS-epoch_590.onnx",
                  "BiRefNet-v2-onnx/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
                  "BiRefNet-v2-onnx/BiRefNet-general-epoch_244.onnx",
                  "BiRefNet-v2-onnx/BiRefNet-general-resolution_512x512-fp16-epoch_216.onnx",
                  "BiRefNet-v2-onnx/BiRefNet-HRSOD_DHU-epoch_115.onnx",
                  "BiRefNet-v2-onnx/BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx",
                  "BiRefNet-v2-onnx/BiRefNet-matting-epoch_100.onnx",
                  "BiRefNet-v2-onnx/BiRefNet-portrait-epoch_150.onnx",],
    "Depth-Anything-2-Onnx" : ["Depth-Anything-2-Onnx/depth_anything_v2_vitb.onnx",
                  "Depth-Anything-2-Onnx/depth_anything_v2_vitl.onnx",
                  "Depth-Anything-2-Onnx/depth_anything_v2_vits.onnx",
                  "depth-pro-onnx/depth_pro.onnx",],
    "dwpose-onnx" : ["dwpose-onnx/yolox_l.onnx",
                     "dwpose-onnx/dw-ll_ucoco_384.onnx"],
    "yolo-nas-pose-onnx" : ["yolo-nas-pose-onnx/yolox_l_dynamic_batch_opset_17_sim.onnx",
                  "yolo-nas-pose-onnx/yolo_nas_pose_l_0.1.onnx",
                  "yolo-nas-pose-onnx/yolo_nas_pose_l_0.2.onnx",
                  "yolo-nas-pose-onnx/yolo_nas_pose_l_0.5.onnx",
                  "yolo-nas-pose-onnx/yolo_nas_pose_l_0.8.onnx",
                  "yolo-nas-pose-onnx/yolo_nas_pose_l_0.35.onnx",],
    "facerestore-onnx" : ["facerestore-onnx/codeformer.onnx",
                        "facerestore-onnx/gfqgan.onnx",],
    "rife-onnx" : ["rife-onnx/rife47_ensemble_True_scale_1_sim.onnx",
                  "rife-onnx/rife48_ensemble_True_scale_1_sim.onnx",
                  "rife-onnx/rife49_ensemble_True_scale_1_sim.onnx",],
    "Upscaler-Onnx" : ["Upscaler-Onnx/4x_foolhardy_Remacri.onnx",
                  "Upscaler-Onnx/4x_NMKD-Siax_200k.onnx",
                  "Upscaler-Onnx/4x_RealisticRescaler_100000_G.onnx",
                  "Upscaler-Onnx/4x-AnimeSharp.onnx",
                  "Upscaler-Onnx/4x-UltraSharp.onnx",
                  "Upscaler-Onnx/4x-WTP-UDS-Esrgan.onnx",
                  "Upscaler-Onnx/RealESRGAN_x4.onnx",],
}

model_SHA256 = {
                'BiRefNet-v2-onnx/BiRefNet_lite-general-2K-epoch_232.onnx': "", 
                'BiRefNet-v2-onnx/BiRefNet-COD-epoch_125.onnx': "", 
                'BiRefNet-v2-onnx/BiRefNet-DIS-epoch_590.onnx': "", 
                'BiRefNet-v2-onnx/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx': "", 
                'BiRefNet-v2-onnx/BiRefNet-general-epoch_244.onnx': "", 
                'BiRefNet-v2-onnx/BiRefNet-general-resolution_512x512-fp16-epoch_216.onnx': "", 
                'BiRefNet-v2-onnx/BiRefNet-HRSOD_DHU-epoch_115.onnx': "", 
                'BiRefNet-v2-onnx/BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx': "", 
                'BiRefNet-v2-onnx/BiRefNet-matting-epoch_100.onnx': "", 
                'BiRefNet-v2-onnx/BiRefNet-portrait-epoch_150.onnx': "", 
                'Depth-Anything-2-Onnx/depth_anything_v2_vitb.onnx': "", 
                'Depth-Anything-2-Onnx/depth_anything_v2_vitl.onnx': "", 
                'Depth-Anything-2-Onnx/depth_anything_v2_vits.onnx': "", 
                'depth-pro-onnx/depth_pro.onnx': "", 
                'dwpose-onnx/yolox_l.onnx': "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411", 
                'dwpose-onnx/dw-ll_ucoco_384.onnx': "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843", 
                'yolo-nas-pose-onnx/yolox_l_dynamic_batch_opset_17_sim.onnx': "", 
                'yolo-nas-pose-onnx/yolo_nas_pose_l_0.1.onnx': "", 
                'yolo-nas-pose-onnx/yolo_nas_pose_l_0.2.onnx': "", 
                'yolo-nas-pose-onnx/yolo_nas_pose_l_0.5.onnx': "", 
                'yolo-nas-pose-onnx/yolo_nas_pose_l_0.8.onnx': "", 
                'yolo-nas-pose-onnx/yolo_nas_pose_l_0.35.onnx': "", 
                'facerestore-onnx/codeformer.onnx': "", 
                'facerestore-onnx/gfqgan.onnx': "", 
                'rife-onnx/rife47_ensemble_True_scale_1_sim.onnx': "", 
                'rife-onnx/rife48_ensemble_True_scale_1_sim.onnx': "", 
                'rife-onnx/rife49_ensemble_True_scale_1_sim.onnx': "", 
                'Upscaler-Onnx/4x_foolhardy_Remacri.onnx': "", 
                'Upscaler-Onnx/4x_NMKD-Siax_200k.onnx': "", 
                'Upscaler-Onnx/4x_RealisticRescaler_100000_G.onnx': "", 
                'Upscaler-Onnx/4x-AnimeSharp.onnx': "", 
                'Upscaler-Onnx/4x-UltraSharp.onnx': "", 
                'Upscaler-Onnx/4x-WTP-UDS-Esrgan.onnx': "", 
                'Upscaler-Onnx/RealESRGAN_x4.onnx': ""
                }

model_class = {"BiRefNet-v2-onnx":"BiRefNet",
               "Depth-Anything-2-Onnx":"Depth-Anything",
               "dwpose-onnx":"dwpose",
               "yolo-nas-pose-onnx" :"yolo-nas-pose",
               "facerestore-onnx":"facerestore",
               "rife-onnx":"rife",
               "Upscaler-Onnx":"upscaler"}

download_path_onnx = os.path.join(folder_paths.models_dir,"tensorrt/TensorRT-ONNX-collect")

CATEGORY_NAME = "TensoRT/plug-in"

# 目前仅支持linux
class building_tensorrt_engine:
    @classmethod
    def INPUT_TYPES(cls):
        all_model_name = []
        for i in model_collect.values():
            all_model_name = all_model_name + i

        # 区分本地模型，待完成
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
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "building"
    CATEGORY = CATEGORY_NAME
    def building(self,select_model,force_building,use_fp16,MirrorDownload):
        # 准备路径和文件字符串
        onnx_name = os.path.basename(select_model)
        trt_name = os.path.splitext(onnx_name)[0] + ".engine"
        new_path = os.path.join(folder_paths.models_dir,"tensorrt",model_class[select_model.split("/")[0]])
        if not os.path.isdir(new_path): os.mkdir(new_path) #创建输出路径
        trt_path = os.path.join(new_path,trt_name)
        onnx_path = os.path.join(download_path_onnx,select_model)
        
        model = None
        if not os.path.isfile(trt_path):
            # 若本地没有模型则下载
            if not os.path.isfile(onnx_path):
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
                
            print("Download completed, Start Conversion...")
            # 开始转换
            try:
                if select_model == "rife-onnx": model = self.building_4(onnx_path, trt_path, use_fp16)
                elif select_model == "BiRefNet-v2-onnx": model = self.building_RGB(onnx_path, trt_path)
                else : model = self.building_A(onnx_path, trt_path, use_fp16)
                print(f"Conversion successful! The model is saved in:\n {trt_path}")
            except:
                print("Conversion failed !")

        # 删除后重新转换，暂时不用
        elif force_building:
            # os.remove(trt_path)
            print(f"{trt_path} \n Already exists, skip conversion !")
            pass

        return (trt_path,)
        
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

    def building_RGB(self, onnx_path, trt_path):
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

    def building_4(self, onnx_path, trt_path, use_fp16):
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



NODE_CLASS_MAPPINGS = {
    "building_tensorrt_engine":building_tensorrt_engine,
}
