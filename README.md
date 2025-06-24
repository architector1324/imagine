## ✨ Imagine: A Lightweight Stable Diffusion Text-to-Image API

![](logo.png)

**Imagine** is a simple yet powerful HTTP server designed for generating images from text prompts using Stable Diffusion models and the Hugging Face `diffusers` library. It provides a single, straightforward endpoint that takes a JSON request and returns the generated image as a base64-encoded string, perfect for quick integration into your applications.

### Key Features

*   **Minimalist API:** One easy-to-use `/generate` endpoint for all your text-to-image needs.
*   **Text-to-Image (txt2img) Only:** Focused on the core functionality, no `img2img` or other complex modes, ensuring a lean and focused codebase.
*   **Base64 Output:** Seamlessly receive generated images as base64 strings, making them easy to embed directly into web pages, mobile apps, or other services without managing file storage.
*   **Configurable Parameters:** Control image generation parameters like `width`, `height`, `num_steps`, `guidance_scale`, `sampler`, `seed`, and `negative_prompt` via the JSON payload.
*   **Built with Flask & Diffusers:** Leveraging robust and popular Python libraries for reliability and ease of use.
*   **Hardware Agnostic:** Supports CPU, CUDA (NVIDIA), MPS (Apple Silicon), and potentially ROCm (AMD) depending on your PyTorch setup.

### Why Imagine?

**Imagine** is inspired by the philosophy of tools like **Ollama** and offers a similar approach for Stable Diffusion models. It is ideal for developers needing a lightweight solution to run Stable Diffusion image generation **as a local background service**.

This enables integration from **anywhere**: command-line scripts, Python applications, web pages, or other services. You no longer need to worry about the overhead and complexity of more feature-rich diffusion UIs; Imagine provides fast, API-driven access to Stable Diffusion, designed for seamless deployment and easy consumption.

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
    (For CUDA (NVIDIA GPU) usage, ensure `torch` is correctly installed, e.g.: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)

3.  **Specify your Stable Diffusion model:**
    Edit `imagine.py` and set `DEFAULT_MODEL` to the path of your `.safetensors` or `.ckpt` model file (e.g., `dreamshaper_8.safetensors`).

4.  **Configure Device (for performance):**
    In `imagine.py`, change `DEFAULT_DEVICE = 'cpu'` to `'cuda'` (NVIDIA), `'mps'` (Apple Silicon), or leave as `'cpu'` for CPU-only inference (slower).

5.  **Run the server:**
    ```bash
    python imagine.py
    ```
    The server will start on `http://0.0.0.0:5000/`.

6.  **Send a POST request to `/generate`:**

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
         http://localhost:5000/generate | jq .
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

## Advanced Deployment: Running as a Systemd Service

For persistent and reliable operation, you can set up **Imagine** as a `systemd` service. This ensures the server starts automatically on boot and restarts in case of crashes.

**1. Create Symlinks and Ensure Executability:**

Ensure both your server script (`imagine.py`) and CLI utility script (`imagine-cli.py`) are executable and symlinked to a common `PATH` directory like `/usr/bin/`.

```bash
# Make the scripts executable
chmod +x /path/to/imagine/imagine.py
chmod +x /path/to/imagine/imagine-cli.py

# Create symlinks
# Replace `/path/to/imagine` with the actual path to your project directory if different
sudo ln -s /path/to/imagine/imagine.py /usr/bin/imagine
sudo ln -s /path/to/imagine/imagine-cli.py /usr/bin/imagine-cli
```

**2. Create the Systemd Service File:**

Create a file named `imagine.service` in `/etc/systemd/system/`:

```bash
sudo nano /etc/systemd/system/imagine.service
```

Paste the following content into the file:

```ini
[Unit]
Description=Imagine: Stable Diffusion Image Generation Server
After=network.target syslog.target

[Service]
# REPLACE 'arch' WITH YOUR LINUX USERNAME!
# Run the service as your current user.
# This simplifies permissions as the script and model are likely in your home directory.
User=arch

# The command to execute when the service starts.
# Ensure /usr/bin/imagine points to your main server script (imagine.py).
ExecStart=/usr/bin/imagine

# Restart the service if it crashes
Restart=on-failure
RestartSec=5s

# Standard output and error will be directed to the systemd journal for easy debugging
StandardOutput=journal
StandardError=journal

# Type of service: simple (default) or forking
Type=simple

[Install]
# This unit should be started when the system reaches multi-user.target (normal boot)
WantedBy=multi-user.target
```

**Important Notes:**
*   **Replace `arch` with your actual Linux username!**
*   Ensure `/usr/bin/imagine` symlink correctly points to your **server script** (`imagine.py`).

**3. Enable and Start the Service:**

After saving the `imagine.service` file:

```bash
# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable imagine.service

# Start the service immediately
sudo systemctl start imagine.service
```

**4. Check Service Status and Logs:**

To verify that the service is running correctly:

```bash
sudo systemctl status imagine.service
```

To view real-time logs for debugging:

```bash
journalctl -u imagine.service -f
```