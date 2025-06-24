## ✨ Imagine: A Lightweight Stable Diffusion Text-to-Image API

![](logo.png)

**Imagine** is a simple yet powerful HTTP server designed for generating images from text prompts using Stable Diffusion models and the Hugging Face `diffusers` library. It provides a single, straightforward endpoint that takes a JSON request and returns the generated image as a base64-encoded string, perfect for quick integration into your applications.

### Key Features

*   **Minimalist API:** One easy-to-use `/generate_image` endpoint for all your text-to-image needs.
*   **Text-to-Image (txt2img) Only:** Focused on the core functionality, no `img2img` or other complex modes, ensuring a lean and focused codebase.
*   **Base64 Output:** Seamlessly receive generated images as base64 strings, making them easy to embed directly into web pages, mobile apps, or other services without managing file storage.
*   **Configurable Parameters:** Control image generation parameters like `width`, `height`, `num_steps`, `guidance_scale`, `sampler`, `seed`, and `negative_prompt` via the JSON payload.
*   **Efficient Model Loading:** Your Stable Diffusion model is loaded once into memory when the server starts, ensuring fast subsequent image generations.
*   **Built with Flask & Diffusers:** Leveraging robust and popular Python libraries for reliability and ease of use.
*   **Hardware Agnostic:** Supports CPU, CUDA (NVIDIA), MPS (Apple Silicon), and potentially ROCm (AMD) depending on your PyTorch setup.

### Why Imagine?

Imagine is ideal for developers who need to quickly add Stable Diffusion image generation capabilities to their backend services, web applications, or internal tools without the overhead and complexity of more feature-rich diffusion UIs. It's designed for rapid deployment and easy consumption.

### Setup & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/imagine.git
    cd imagine
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch diffusers transformers accelerate flask Pillow argparse requests base64
    ```
    (Ensure your `torch` installation is compatible with your hardware, e.g., for CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)

3.  **Place your Stable Diffusion model:**
    Update the `DEFAULT_MODEL` variable in `imagine.py` to point to your `.safetensors` or `.ckpt` model file (e.g., `dreamshaper_8.safetensors`).

4.  **Configure Device (Crucial for Performance!):**
    Edit `imagine.py` and change `DEFAULT_DEVICE = 'cpu'` to `'cuda'` (for NVIDIA GPUs), `'mps'` (for Apple Silicon M-series chips), or leave as `'cpu'` for CPU-only inference (slower).

5.  **Run the server:**
    ```bash
    python imagine.py
    ```
    The server will start on `http://0.0.0.0:5000/`.

6.  **Send a POST request to `/generate_image`:**

    **Example using `curl`:**
    ```bash
    curl -X POST -H "Content-Type: application/json" \
         -d '{
             "model": "<path to>/dreamshaper_8.safetensors"
             "prompt": "a photo of an astronaut riding a horse on mars, epic, cinematic, detailed",
             "width": 768,
             "height": 512,
             "num_steps": 25,
             "guidance": 7.0,
             "sampler": "DPM++ 2M",
             "neg": "ugly, deformed, blurry, low quality"
         }' \
         http://localhost:5000/generate_image | jq .
    ```

    The response will be a JSON object containing the `img` string and the `seed`.

### Imagine CLI

### Description

A simple command-line script for generating images based on a text prompt using the `diffusers` library.

### Usage

```
usage: imagine-cli.py [-m MODEL] [-o OUTPUT] [-w WIDTH] [-h HEIGHT] [-n NUM_STEPS] [-g GUIDANCE] [-s SAMPLER] [--seed SEED] [--neg NEG] [--help] prompt [prompt ...]

SD image generator

positional arguments:
  prompt                Prompt for model

options:
  -m, --model MODEL     SD model
  -o, --output OUTPUT   Output image
  -w, --width WIDTH     Output image width
  -h, --height HEIGHT   Output image height
  -n, --num_steps NUM_STEPS
                        Number of steps
  -g, --guidance GUIDANCE
                        Guidance scale
  -s, --sampler SAMPLER
                        SD Sampler ['DDIM', 'Euler', 'Euler a', 'Heun', 'LMS', 'DPM++ 2M', 'DPM++ 2S', 'DPM++ SDE', 'DPM2', 'DPM2 a']
  --seed SEED           Seed
  --neg NEG             Negative prompt
  --help
```

#### Example Usage

```bash
./imagine-cli.py 'a photo of an astronaut riding a horse on mars, epic, cinematic, detailed' -w 768 -h 512 -n 25 -g 7.0 -s 'DPM++ 2M' --neg 'ugly, deformed, blurry, low quality'
```

### Output

The script will save the generated image to the current directory with a filename based on the prompt or a timestamp.
