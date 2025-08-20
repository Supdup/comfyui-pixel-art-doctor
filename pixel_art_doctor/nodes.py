import torch
import collections
from PIL import Image
import torchvision.transforms.functional as TF

class PixelArtDoctor:
    """
    Takes a 'fake' pixel art image and resamples it down to its true native resolution.
    This version uses a custom dominant color sampling method for a perfectly sharp result.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_threshold": ("FLOAT", { "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01, "display": "slider" }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "detected_pixel_size")
    FUNCTION = "process"
    CATEGORY = "Image Processing"

    def tensor_to_pil(self, tensor_image: torch.Tensor) -> Image.Image:
        """Converts a single image tensor (H, W, C) to a PIL Image."""
        return TF.to_pil_image(tensor_image.permute(2, 0, 1))

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Converts a PIL Image to a single image tensor (H, W, C)."""
        return TF.to_tensor(pil_image).permute(1, 2, 0)

    def find_pixel_size_torch(self, pil_image: Image.Image, threshold: float, device: torch.device):
        """Analyzes the image to find the average pixel block size using PyTorch."""
        gray_tensor = TF.to_tensor(pil_image.convert('L')).squeeze(0).to(device)
        diff_x = torch.abs(torch.diff(gray_tensor, dim=1)); diff_y = torch.abs(torch.diff(gray_tensor, dim=0))
        x_projection = torch.sum(diff_x, dim=0); y_projection = torch.sum(diff_y, dim=1)

        def get_most_common_distance(projection: torch.Tensor, threshold_val: float) -> int:
            if torch.max(projection) > 0: projection = projection / torch.max(projection)
            peaks = torch.where(projection > threshold_val)[0]
            if len(peaks) < 2: return 8
            distances = torch.diff(peaks); distances = distances[distances > 1]
            if len(distances) < 1: return 8
            return collections.Counter(distances.cpu().tolist()).most_common(1)[0][0]

        return get_most_common_distance(x_projection, threshold), get_most_common_distance(y_projection, threshold)

    def resample_by_mode(self, image: Image.Image, target_width: int, target_height: int, block_w: int, block_h: int) -> Image.Image:
        """Resamples the image by finding the most common color in each source block."""
        new_image = Image.new('RGB', (target_width, target_height))

        for y in range(target_height):
            for x in range(target_width):
                # Define the box in the original image
                left = round(x * block_w)
                top = round(y * block_h)
                right = round((x + 1) * block_w)
                bottom = round((y + 1) * block_h)

                box = (left, top, right, bottom)

                # Get all pixels from the box
                source_block = image.crop(box)
                pixels = list(source_block.getdata())

                if not pixels: continue

                # Find the most common color (the mode)
                most_common_color = collections.Counter(pixels).most_common(1)[0][0]
                new_image.putpixel((x, y), most_common_color)

        return new_image


    def process(self, image: torch.Tensor, detection_threshold: float):
        processed_images = []

        first_pil_image = self.tensor_to_pil(image[0])
        avg_pixel_width, avg_pixel_height = self.find_pixel_size_torch(first_pil_image, detection_threshold, image.device)
        print(f"[PixelArtResampler] Detected pixel size: {avg_pixel_width}w x {avg_pixel_height}h.")

        for img_tensor in image:
            pil_image = self.tensor_to_pil(img_tensor)

            # Calculate the target native resolution
            native_width = round(pil_image.width / avg_pixel_width)
            native_height = round(pil_image.height / avg_pixel_height)
            if native_width < 1 or native_height < 1:
                native_width, native_height = pil_image.width // 8, pil_image.height // 8

            # --- Use the new dominant color resampling method ---
            native_resolution_image = self.resample_by_mode(
                pil_image,
                native_width,
                native_height,
                avg_pixel_width,
                avg_pixel_height
            )

            processed_images.append(self.pil_to_tensor(native_resolution_image).to(image.device))

        output_tensor = torch.stack(processed_images)

        detected_pixel_size = round((avg_pixel_width + avg_pixel_height) / 2)

        return (output_tensor, detected_pixel_size)
