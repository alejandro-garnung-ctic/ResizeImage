import torch
from comfy.utils import ProgressBar
from PIL import Image
import numpy as np

class ResizeImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": [
                ("image", "IMAGE"),
                ("width", "INT", {"default": 512, "min": 1, "max": 8192}),
                ("height", "INT", {"default": 512, "min": 1, "max": 8192})
            ]
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize"
    CATEGORY = "image/resizer"

    def resize(self, image, width, height):
        # Convertir imagen de tensor a formato PIL
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        img = img.resize((width, height), Image.LANCZOS)
        
        # Reconvertir a tensor
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0) # Añadir batch dimension
        
        return (img_tensor,)