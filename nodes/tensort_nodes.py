import glob
import os
import time

from typing import List, Union
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

import comfy.model_management as mm
import folder_paths

from .models_BiRefNet.models.birefnet import BiRefNet
from .export_trt.trt_utilities import Engine_4
from .export_trt.utilities import Engine


folder_paths.add_model_folder_path("BiRefNet",os.path.join(folder_paths.models_dir, "BiRefNet"))
device = "cuda" if torch.cuda.is_available() else "cpu"
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
    "dwpose-onnx" : ["dwpose-onnx/yolox_l_dynamic_batch_opset_17_sim.onnx",
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

model_class = {"BiRefNet-v2-onnx":"BiRefNet",
               "Depth-Anything-2-Onnx":"Depth-Anything",
               "dwpose-onnx":"dwpose",
               "facerestore-onnx":"facerestore",
               "rife-onnx":"rife",
               "Upscaler-Onnx":"upscaler"}


CATEGORY_NAME = "TensoRT/plug-in"

# 目前仅支持linux
class building_tensorrt_engine:
    @classmethod
    def INPUT_TYPES(cls):
        model_class_name = list(model_collect.keys())
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
                "select_class":(model_class_name,{"default": model_class_name[0],}),
                "select_model": (all_model_name,{"default": all_model_name[0],}),
                "force_building": ("BOOLEAN",{"default": False}),
                "use_fp16": ("BOOLEAN",{"default": True}),
                "MirrorDownload": ("BOOLEAN",{"default": True}),
            }
        }

    RETURN_TYPES = ("trt_model",)
    OUTPUT_NODE = True
    FUNCTION = "building"
    CATEGORY = CATEGORY_NAME
    def building(self,select_class,select_model,force_building,use_fp16,MirrorDownload):
        # 准备路径和文件字符串
        onnx_name = os.path.basename(select_model)
        trt_name = os.path.splitext(onnx_name)[0] + ".engine"
        new_path = os.path.join(folder_paths.models_dir,"tensorrt",model_class[select_class])
        if not os.path.isdir(new_path): os.mkdir(new_path) #创建
        trt_path = os.path.join(new_path,trt_name)
        onnx_path = os.path.join(folder_paths.models_dir,"tensorrt/TensorRT-ONNX-collect",select_model)
        
        model = None
        if not os.path.isfile(trt_path):
            # 若本地没有模型则下载
            if not os.path.isfile(onnx_path):
                from huggingface_hub import hf_hub_download
                if MirrorDownload: os.system("export HF_ENDPOINT='https://hf-mirror.com'")
                hf_hub_download(repo_id= "EmmaJohnson311/TensorRT-ONNX-collect",
                                filename = select_model,
                                local_dir = onnx_path
                                )

            # 开始转换
            if select_model == "rife-onnx": model = self.building_4(onnx_path, trt_path, use_fp16)
            elif select_model == "BiRefNet-v2-onnx": model = self.building_RGB(onnx_path, trt_path)
            else : model = self.building_A(onnx_path, trt_path, use_fp16)

        # 删除后重新转换，暂时不用
        elif force_building:
            # os.remove(trt_path)
            print(f"{trt_path}已存在，跳过转换")
            pass
        return (None,)
        
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


class load_BiRefNet2_General:
    def model_name(self):
        self.pretrained_weights = [
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

    @classmethod
    def INPUT_TYPES(cls):
        local_models= folder_paths.get_filename_list("BiRefNet"),
        if isinstance(local_models,tuple):
            local_models = list(local_models[0])
        local_models.append(os.listdir(os.path.join(folder_paths.models_dir,"tensorrt/BiRefNet")))
        return {
            "required": {
                "birefnet_model": (local_models,{"default": local_models[0],}),
            }
        }

    RETURN_TYPES = ("BRNMODEL",)
    RETURN_NAMES = ("birefnet",)
    FUNCTION = "load_model"
    CATEGORY = CATEGORY_NAME
  
    def load_model(self,birefnet_model):
        model_path = folder_paths.get_full_path("BiRefNet", birefnet_model)
        if not os.path.isfile(model_path): 
            model_path = os.path.join(folder_paths.models_dir,"tensorrt/BiRefNet",birefnet_model)
        print(f"load model: {model_path}")

        if birefnet_model.endswith('.onnx'):
                import onnxruntime
                providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
                #model_path = folder_paths.get_full_path("BiRefNet", birefnet_model)
                onnx_session = onnxruntime.InferenceSession(
                    model_path,
                    providers=providers
                )
                return (('onnx',onnx_session),),
        elif birefnet_model.endswith('.engine') or birefnet_model.endswith('.trt') or birefnet_model.endswith('.plan'):
            #model_path = folder_paths.get_full_path("BiRefNet", birefnet_model)
            import tensorrt as trt
            # 创建logger：日志记录器
            logger = trt.Logger(trt.Logger.WARNING)
            # 创建runtime并反序列化生成engine
            with open(model_path ,'rb') as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            return (('tensorrt',engine),)
        else :
            raise TypeError("Only supports  .onnx  .engine  .trt  .plan")

class BiRefNet2_tensort:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "birefnet": ("BRNMODEL",),
                "image": ("IMAGE",),
                "reversal_mask": ("BOOLEAN",{"default":False})
            }
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask", )
    FUNCTION = "remove_background"
    CATEGORY = CATEGORY_NAME
  
    def remove_background(self, birefnet, image,reversal_mask):
        net_type, net = birefnet
        processed_masks = []

        transform_image = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        for image in image:
            orig_image = tensor2pil(image)
            w,h = orig_image.size
            image = self.resize_image(orig_image)
            im_tensor = transform_image(image).unsqueeze(0)
            im_tensor=im_tensor.to(device)
            if net_type=='onnx':
                input_name = net.get_inputs()[0].name
                input_images_numpy = tensor2np(im_tensor)
                result = torch.tensor(
                    net.run(None, {input_name: input_images_numpy if device == 'cpu' else input_images_numpy})[-1]
                ).squeeze(0).sigmoid().cpu()
            
            elif net_type=='tensorrt':
                from .models_BiRefNet import common
                with net.create_execution_context() as context:
                    image_data = np.expand_dims(transform_image(orig_image), axis=0).ravel()
                    engine = net
                    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
                    np.copyto(inputs[0].host, image_data)
                    trt_outputs = common.do_inference(context, engine, bindings, inputs, outputs, stream)
                   
                    numpy_array = np.array(trt_outputs[-1].reshape((1, 1, 1024, 1024)))
                    result = torch.from_numpy(numpy_array).sigmoid().cpu()
                    common.free_buffers(inputs, outputs, stream)
            else:
                with torch.no_grad():
                    result = net(im_tensor)[-1].sigmoid().cpu()
                    
                    
            result = torch.squeeze(F.interpolate(result, size=(h,w)))
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)
            result = torch.cat(result, dim=0)
            processed_masks.append(result)

        new_masks = torch.cat(processed_masks, dim=0)
        if reversal_mask : new_masks = 1 - new_masks
        return (new_masks,)
    
    def resize_image(self,image):
        image = image.convert('RGB')
        model_input_size = (1024, 1024)
        image = image.resize(model_input_size, Image.BILINEAR)
        return image
    

NODE_CLASS_MAPPINGS = {
    "building_tensorrt_engine":building_tensorrt_engine,
    "load_BiRefNet2_General": load_BiRefNet2_General,
    'BiRefNet2_tensort':BiRefNet2_tensort
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "building_tensorrt_engine":"building tensorrt engine",
    "load_BiRefNet2_General": "load BiRefNet2 General",
    "BiRefNet2_tensort": "BiRefNet2 tensort",
}
