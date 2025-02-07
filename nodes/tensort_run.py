import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import folder_paths
from comfy.utils import ProgressBar

from .dwpose import DWposeDetector
from .tensort_convert import model_class

folder_paths.add_model_folder_path("BiRefNet",os.path.join(folder_paths.models_dir, "BiRefNet"))
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision(["high", "highest"][0])

CATEGORY_NAME = "TensoRT/plug-in"
model_Warning = "* Please convert the model first ! *"

trt_path_dict = {}
for k,v in model_class.items():
    trt_path = os.path.join(folder_paths.models_dir, "tensorrt", v)
    if not os.path.isdir(trt_path):
        os.makedirs(trt_path)
    trt_path_dict[v] = trt_path

class load_dwpos_model:
    dwpos_path = os.path.join(folder_paths.models_dir, "tensorrt", "dwpose")
    @classmethod
    def INPUT_TYPES(s):
        yolox_l_path = os.listdir(trt_path_dict["dwpose"])
        if len(yolox_l_path) == 0:
            yolox_l_path = [model_Warning]
        return {
            "required": {
                "yolox_l": (yolox_l_path, {"default":yolox_l_path[0]}),
                "ll_ucoco_384": (yolox_l_path, {"default":yolox_l_path[-1]}),
            }
        }
    RETURN_TYPES = ("dwpose_model",)
    FUNCTION = "main"
    CATEGORY = CATEGORY_NAME
    def main(self,yolox_l,ll_ucoco_384):
        yolox_l = os.path.join(self.__class__.dwpos_path,yolox_l)
        ll_ucoco_384 = os.path.join(self.__class__.dwpos_path,ll_ucoco_384)
        dwpose = DWposeDetector(yolox_l,ll_ucoco_384)
        return (dwpose,)

class Dwpose_Tensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dwpose": ("dwpose_model",),
                "images": ("IMAGE",),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "show_body": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = CATEGORY_NAME

    def main(self, dwpose, images, show_face, show_hands, show_body):

        pbar = ProgressBar(images.shape[0])
        pose_frames = []

        for img in images:
            img_np_hwc = (img.cpu().numpy() * 255).astype(np.uint8)
            result = dwpose(image_np_hwc=img_np_hwc, show_face=show_face,
                            show_hands=show_hands, show_body=show_body)
            pose_frames.append(result)
            pbar.update(1)

        pose_frames_np = np.array(pose_frames).astype(np.float32) / 255
        return (torch.from_numpy(pose_frames_np),)


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
        local_models = os.listdir(trt_path_dict["BiRefNet"])
        if len(local_models) == 0:
            local_models = model_Warning
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
    "load_dwpos_model": load_dwpos_model,
    "Dwpose_Tensorrt": Dwpose_Tensorrt,
    "load_BiRefNet2_General":load_BiRefNet2_General,
    "BiRefNet2_tensort":BiRefNet2_tensort,
}