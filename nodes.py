import sys
import os
# Get the absolute path of the parent directory of the current script
my_dir = os.path.dirname(os.path.abspath(__file__))

# Add the My directory path to the sys.path list
sys.path.append(my_dir)
sys.path.append(my_dir+"/module")

from torchvision.transforms import ToPILImage,ToTensor
from module.idm_vton import start_tryon
import torch

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
                "human_image": ("IMAGE",),
                "clothe_image": ("IMAGE",),
                "densepose_image": ("IMAGE",),
                "auto_mask_category": (["upper_body", "lower_body", "dresses"],),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_and_execute"

    CATEGORY = "IDM-VTON"

    def process_and_execute(self, human_image, clothe_image, densepose_image, auto_mask_category, prompt):
        to_pil = ToPILImage()
        to_tensor = ToTensor()

        human_image_batch = human_image.movedim(-1,1)
        clothe_image_batch = clothe_image.movedim(-1,1)
        densepose_image_batch = densepose_image.movedim(-1,1)
        image_out_list = []
        for human_img, clothe_img, densepose_img in zip(human_image_batch, clothe_image_batch, densepose_image_batch):
            human_pil = to_pil(human_img)
            clothe_pil = to_pil(clothe_img)
            densepose_pil = to_pil(densepose_img)

            dict = {}
            dict["background"] = human_pil
            image_out, masked_img = start_tryon(dict, clothe_pil, prompt, densepose_pil, None, auto_mask_category)
            image_out_list.append(image_out)

        imgs = [to_tensor(img) for img in image_out_list]
        tensor_batch = torch.stack(imgs, dim=0)

        return (tensor_batch.movedim(1, -1),)

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
                "human_image": ("IMAGE",),
                "clothe_image": ("IMAGE",),
                "densepose_image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_and_execute"

    CATEGORY = "IDM-VTON"

    def process_and_execute(self, human_image, clothe_image, densepose_image, mask_image, prompt):
        to_pil = ToPILImage()
        to_tensor = ToTensor()

        human_image_batch = human_image.movedim(-1,1)
        clothe_image_batch = clothe_image.movedim(-1,1)
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
            image_out, masked_img = start_tryon(dict, clothe_pil, prompt, densepose_pil, mask_pil)
            image_out_list.append(image_out)

        imgs = [to_tensor(img) for img in image_out_list]
        tensor_batch = torch.stack(imgs, dim=0)

        return (tensor_batch.movedim(1, -1),)

NODE_CLASS_MAPPINGS = {
    "IDM_VTON_NN": IDM_VTON_Processor,
    "IDM_VTON_MASK_NN": IDM_VTON_MASK_Processor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IDM_VTON_NN": "IDM-VTON (diffusers)",
    "IDM_VTON_MASK_NN": "IDM-VTON with Mask (diffusers)",
}
