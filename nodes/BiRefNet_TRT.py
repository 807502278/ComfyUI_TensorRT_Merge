import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
import comfy.model_management as mm
import os
import folder_paths
from .BiRefNet.models.birefnet import BiRefNet
from typing import List, Union

## ComfyUI portable standalone build for Windows 
# model_path = os.path.join(current_path, "ComfyUI"+os.sep+"models"+os.sep+"BiRefNet")

folder_paths.add_model_folder_path("BiRefNet",os.path.join(folder_paths.models_dir, "BiRefNet"))

CATEGORY_NAME = "TensoRT/plug-in"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision(["high", "highest"][0])

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Tensor to np
def tensor2np(tensor: torch.Tensor) -> List[np.ndarray]:
  if len(tensor.shape) == 3:  # Single image
    return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
  else:  # Batch of images
    return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]

# https://objects.githubusercontent.com/github-production-release-asset-2e65be/525717745/81693dcf-8d42-4ef6-8dba-1f18f87de174?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20241014%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241014T003944Z&X-Amz-Expires=300&X-Amz-Signature=ec867061341cf6498cf5740c36f49da22d4d3d541da48d6e82c7bce0f3b63eaf&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3DBiRefNet-COD-epoch_125.pth&response-content-type=application%2Foctet-stream

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

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
colors = ["transparency", "green", "white", "red", "yellow", "blue", "black", "pink", "purple", "brown", 
          "violet", "wheat", "whitesmoke", "yellowgreen", "turquoise", "tomato", "thistle", "teal", "tan", 
          "steelblue", "springgreen", "snow", "slategrey", "slateblue", "skyblue", "orange"]

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image

class BiRefNet_ModelLoader_TRT:
    @classmethod
    def INPUT_TYPES(cls):
        local_models= folder_paths.get_filename_list("BiRefNet"),
        if isinstance(local_models,tuple): local_models = list(local_models[0]) #转列表
        return {
            "required": {
                "load_mode": (['local', 'local_onnx', 'local_trt', 'pretrained'], {"default": "local"}),
                "birefnet_model": (local_models, {
                    "default": local_models[0],
                    "pysssss.binding": [{
                        "source": "load_mode",
                        "callback": [
                            {
                                "type": "if",
                                "condition": [
                                    {"left": "$source.value", "op": "eq", "right": '"local"'}
                                ],
                                "true": [
                                    {"type": "set", "target": "$this.options.values", "value": local_models}
                                ],
                                "false": [
                                    {
                                        "type": "if",
                                        "condition": [
                                            {"left": "$source.value", "op": "eq", "right": '"local_onnx"'}
                                        ],
                                        "true": [
                                            {"type": "set", "target": "$this.options.values", "value": local_models}
                                        ],
                                        "false": [
                                            {
                                                "type": "if",
                                                "condition": [
                                                    {"left": "$source.value", "op": "eq", "right": '"local_trt"'}
                                                ],
                                                "true": [
                                                    {"type": "set", "target": "$this.options.values", "value": local_models}
                                                ],
                                                "false": [
                                                    {"type": "set", "target": "$this.options.values", "value": pretrained_weights}
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }]
                })
            }
        }
        return {
            "required": {
                "load_mode": (['local','pretrained'],{"default": "local"}),
                "birefnet_model": (local_models,{
                    "default": local_models[0],
                    "pysssss.binding": [{"source": "load_mode",
                                         "callback": [
                                                    {"type": "if",
                                                     "condition": [{"left": "$source.value",
                                                                    "op": "eq",
                                                                    "right": '"local"'}],
                                                     "true": [{"type": "set",
                                                               "target": "$this.options.values",
                                                               "value": local_models,}],
                                                     "false": [{"type": "set",
                                                                "target": "$this.options.values",
                                                                "value": pretrained_weights,}],
                                                    },]
                                        }]
                                    }),
                
            }
        }

    RETURN_TYPES = ("BRNMODEL",)
    RETURN_NAMES = ("birefnet",)
    FUNCTION = "load_model"
    CATEGORY = CATEGORY_NAME
  
    def load_model(self, load_mode,birefnet_model):
        Accelerator_Model_path = os.path.join(folder_paths.models_dir,"tensorrt")
        if birefnet_model.endswith('.onnx'):
                import onnxruntime
                providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
                model_path = folder_paths.get_full_path("BiRefNet", birefnet_model)
                onnx_session = onnxruntime.InferenceSession(
                    model_path,
                    providers=providers
                )
                return (('onnx',onnx_session),),
        elif birefnet_model.endswith('.engine') or birefnet_model.endswith('.trt') or birefnet_model.endswith('.plan'):
            model_path = folder_paths.get_full_path("BiRefNet", birefnet_model)
            import tensorrt as trt
            # 创建logger：日志记录器
            logger = trt.Logger(trt.Logger.WARNING)
            # 创建runtime并反序列化生成engine
            with open(model_path ,'rb') as f, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            return (('tensorrt',engine),)
            
        else:
            if load_mode =='local':
                net = BiRefNet(bb_pretrained=False)
                model_path = folder_paths.get_full_path("BiRefNet", birefnet_model)
                #print(model_path)
                state_dict = torch.load(model_path, map_location=device)
                unwanted_prefix = '_orig_mod.'
                for k, v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

                net.load_state_dict(state_dict)
                net.to(device)
                net.eval() 
                return (('pytorch',net),)
            else:
                net = BiRefNet.from_pretrained(birefnet_model)
                net.to(device)
                net.eval() 
                return (('pytorch',net),)
class BiRefNet_TRT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "birefnet": ("BRNMODEL",),
                "image": ("IMAGE",),
                "background_color_name": (colors,{"default": "transparency"}),
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "remove_background"
    CATEGORY = CATEGORY_NAME
  
    def remove_background(self, birefnet, image,background_color_name):
        (net_type, net) = birefnet
        processed_images = []
        processed_masks = []
        for image in image:
            orig_image = tensor2pil(image)
            w,h = orig_image.size
            image = resize_image(orig_image)
            im_tensor = transform_image(image).unsqueeze(0)
            im_tensor=im_tensor.to(device)
            if net_type=='onnx':
                input_name = net.get_inputs()[0].name
                input_images_numpy = tensor2np(im_tensor)
                result = torch.tensor(
                    net.run(None, {input_name: input_images_numpy if device == 'cpu' else input_images_numpy})[-1]
                ).squeeze(0).sigmoid().cpu()
            
            elif net_type=='tensorrt':
                from . import common
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
            im_array = (result*255).cpu().data.numpy().astype(np.uint8)
            pil_im = Image.fromarray(np.squeeze(im_array))
            if background_color_name == 'transparency':
                color = (0,0,0,0)
                mode = "RGBA"
            else:
                color = background_color_name
                mode = "RGB"
            new_im = Image.new(mode, pil_im.size, color)
            new_im.paste(orig_image, mask=pil_im)
            new_im_tensor = pil2tensor(new_im)
            pil_im_tensor = pil2tensor(pil_im)
            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)

        return (new_ims, new_masks,)

NODE_CLASS_MAPPINGS = {
    "BiRefNet_ModelLoader_TRT": BiRefNet_ModelLoader_TRT,
    'BiRefNet_TRT':BiRefNet_TRT
}
