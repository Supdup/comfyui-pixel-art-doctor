# ComfyUI Pixel Art Doctor

A ComfyUI custom node that intelligently downscales 'fake' pixel art to its true, native resolution.

This node is the perfect tool for cleaning up pixel art that has been resized, has anti-aliasing, or contains compression artifacts. It automatically detects the grid size and resamples the image for a perfectly sharp, 1:1 pixel result.

---

## ## What It Does

This node solves a common problem where an image looks like pixel art but isn't technically correct. Instead of using a simple resize, which can blur the image, this node:

1.  **Analyzes the image** to automatically detect the size of the "fake" pixel blocks.
2.  **Downscales the image** to its true native resolution.
3.  Uses **dominant color sampling** for each block instead of averaging, which ensures the output is perfectly crisp and sharp with no anti-aliasing or blur.

---

## ## 🚀 Installation

The easiest way to install is by using the [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager).

1.  Open the **ComfyUI Manager** menu.
2.  Click on **Install via Git URL**.
3.  Enter this repository's URL: `https://github.com/Supdup/comfyui-pixel-art-doctor.git`
4.  Click **OK** and restart ComfyUI.

---

## ## How to Use

Load the node by double-clicking the canvas and searching for `Pixel Art Doctor`.

### ### Inputs

- **`image`**: The input 'fake' pixel art image that has artifacts or incorrect scaling.
- **`detection_threshold`**: Controls the sensitivity of the pixel grid detection. The default of `0.1` works for most cases.

### ### Outputs

- **`image`**: The final, downscaled image at its true native resolution. One pixel in this output corresponds to one visual pixel from the input.
- **`detected_pixel_size`**: The average size (in pixels) of the square pixel blocks that were detected in the input image.

### ### Example Workflow

The node outputs a small, native-resolution image. To view it larger, connect its output to an `ImageScale` node and set the `upscale_method` to `nearest-neighbor`.

`(Image Source)` -> `🩺 Pixel Art Doctor` -> `ImageScale (nearest-neighbor)` -> `(Preview Image)`
