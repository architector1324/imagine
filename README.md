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
    pip install torch diffusers transformers accelerate flask Pillow
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

    The response will be a JSON object containing the `image_base64` string and the `seed_used`.
