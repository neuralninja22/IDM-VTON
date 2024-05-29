import sys
import os
# Get the absolute path of the parent directory of the current script
my_dir = os.path.dirname(os.path.abspath(__file__))

# Add the My directory path to the sys.path list
sys.path.append(my_dir)
sys.path.append(my_dir+"/module")

from torchvision.transforms import ToPILImage,ToTensor
from module.idm_vton import start_tryon, create_pipeline
import torch
import numpy as np
from PIL import Image,ImageFilter

def merge_images_with_soft_mask(image1, image2, mask, blur_radius=5):
    mask = mask.convert("L")
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    mask = np.array(mask, dtype=np.float32)

    mask = mask / 255.0

    image1_np = np.array(image1, dtype=np.float32)
    image2_np = np.array(image2, dtype=np.float32)

    result_np = np.empty_like(image1_np)

    for c in range(3):
        result_np[:, :, c] = image1_np[:, :, c] * mask + image2_np[:, :, c] * (1 - mask)

    result_np = np.clip(result_np, 0, 255).astype(np.uint8)
    result = Image.fromarray(result_np, 'RGB')
    return result

class IDM_VTON_PipeLineProcessor:
    """
    IDM-VTON Pipeline
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low_vram": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "process_and_execute"

    CATEGORY = "IDM-VTON"

    def process_and_execute(self, low_vram):
        pipeline = create_pipeline(low_vram)

        return (pipeline,)

class IDM_VTON_Processor:
    """
    IDM-VTON 
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("PIPE_LINE",),
                "human_image": ("IMAGE",),
                "clothes_image": ("IMAGE",),
                "densepose_image": ("IMAGE",),
                "auto_mask_category": (["upper_body", "lower_body", "dresses"],),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "clothes prompt. ex) Short Sleeve Round Neck T-shirts",
                }),
                "fit_human_image_size": ("BOOLEAN", {"default": True}),
                "fix_origin_merge": ("BOOLEAN", {"default": False}),
                "denoise_steps": ("INT", {"default": 30}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("image", "mask_image", "mask_gray_image",)
    FUNCTION = "process_and_execute"

    CATEGORY = "IDM-VTON"

    def process_and_execute(self, pipeline, human_image, clothes_image, densepose_image, auto_mask_category, prompt, fit_human_image_size, fix_origin_merge, denoise_steps, seed):
        to_pil = ToPILImage()
        to_tensor = ToTensor()

        human_image_batch = human_image.movedim(-1,1)
        clothe_image_batch = clothes_image.movedim(-1,1)
        densepose_image_batch = densepose_image.movedim(-1,1)
        image_out_list = []
        masked_img_list = []
        masked_gray_img_list = []
        for human_img, clothe_img, densepose_img in zip(human_image_batch, clothe_image_batch, densepose_image_batch):
            human_pil = to_pil(human_img)
            clothe_pil = to_pil(clothe_img)
            densepose_pil = to_pil(densepose_img)

            dict = {}
            dict["background"] = human_pil
            image_out, masked_img, masked_gray_img = start_tryon(pipeline, dict, clothe_pil, prompt, densepose_pil, None, auto_mask_category, denoise_steps, seed)

            if fit_human_image_size:
                image_out = image_out.resize(human_pil.size)
                masked_img = masked_img.resize(human_pil.size)
                if fix_origin_merge:
                    image_out = merge_images_with_soft_mask(image_out, human_pil, masked_img)

            image_out_list.append(image_out)
            masked_img_list.append(masked_img.convert("RGB"))
            masked_gray_img_list.append(masked_gray_img)

        imgs = [to_tensor(img) for img in image_out_list]
        tensor_batch = torch.stack(imgs, dim=0)
        imgs2 = [to_tensor(img) for img in masked_img_list]
        tensor_batch2 = torch.stack(imgs2, dim=0)
        imgs3 = [to_tensor(img) for img in masked_gray_img_list]
        tensor_batch3 = torch.stack(imgs3, dim=0)

        return (tensor_batch.movedim(1, -1),tensor_batch2.movedim(1, -1),tensor_batch3.movedim(1, -1),)

class IDM_VTON_MASK_Processor:
    """
    IDM-VTON 
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("PIPE_LINE",),
                "human_image": ("IMAGE",),
                "clothes_image": ("IMAGE",),
                "densepose_image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "clothes prompt. ex) Short Sleeve Round Neck T-shirts",
                }),
                "fit_human_image_size": ("BOOLEAN", {"default": True}),
                "fix_origin_merge": ("BOOLEAN", {"default": False}),
                "denoise_steps": ("INT", {"default": 30}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_and_execute"

    CATEGORY = "IDM-VTON"

    def process_and_execute(self, pipeline, human_image, clothes_image, densepose_image, mask_image, prompt, fit_human_image_size, fix_origin_merge, denoise_steps, seed):
        to_pil = ToPILImage()
        to_tensor = ToTensor()

        human_image_batch = human_image.movedim(-1,1)
        clothe_image_batch = clothes_image.movedim(-1,1)
        densepose_image_batch = densepose_image.movedim(-1,1)
        mask_image_batch = mask_image.movedim(-1,1)
        image_out_list = []
        for human_img, clothe_img, densepose_img, mask_img in zip(human_image_batch, clothe_image_batch, densepose_image_batch, mask_image_batch):
            human_pil = to_pil(human_img)
            clothe_pil = to_pil(clothe_img)
            densepose_pil = to_pil(densepose_img)
            mask_pil = to_pil(mask_img)

            dict = {}
            dict["background"] = human_pil
            image_out, masked_img, _ = start_tryon(pipeline, dict, clothe_pil, prompt, densepose_pil, mask_pil, None, denoise_steps, seed)
            if fit_human_image_size:
                image_out = image_out.resize(human_pil.size)
                masked_img = masked_img.resize(human_pil.size)
                if fix_origin_merge:
                    image_out = merge_images_with_soft_mask(image_out, human_pil, masked_img)

            image_out_list.append(image_out)

        imgs = [to_tensor(img) for img in image_out_list]
        tensor_batch = torch.stack(imgs, dim=0)

        return (tensor_batch.movedim(1, -1),)

NODE_CLASS_MAPPINGS = {
    "IDM_VTON_PIPELINE_NN": IDM_VTON_PipeLineProcessor,
    "IDM_VTON_NN": IDM_VTON_Processor,
    "IDM_VTON_MASK_NN": IDM_VTON_MASK_Processor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IDM_VTON_PIPELINE_NN": "IDM-VTON Pipeline (diffusers)",
    "IDM_VTON_NN": "IDM-VTON (diffusers)",
    "IDM_VTON_MASK_NN": "IDM-VTON with Mask (diffusers)",
}
