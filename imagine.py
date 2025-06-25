#!/usr/bin/python

import os
import io
import json
import queue
import torch
import base64
import random
import argparse
import threading
import diffusers

from PIL import Image
from flask import Flask, request, jsonify, Response, stream_with_context

# settings
DEFAULT_MODEL = '/home/arch/AI/models/dreamshaper_8.safetensors'
DEFAULT_FP_PREC = torch.float32
DEFAULT_DEVICE = 'cpu'

os.environ['HF_HUB_OFFLINE'] = '1'

dev = DEFAULT_DEVICE
fp_prec = DEFAULT_FP_PREC

SAMPLERS = {
    'ddim': diffusers.DDIMScheduler,
    'euler': diffusers.EulerDiscreteScheduler,
    'euler a': diffusers.EulerAncestralDiscreteScheduler,
    'heun': diffusers.HeunDiscreteScheduler,
    'lms': diffusers.LMSDiscreteScheduler,
    'dpm++ 2m': diffusers.DPMSolverMultistepScheduler,
    'dpm++ 2s': diffusers.DPMSolverSinglestepScheduler,
    'dpm++ sde': diffusers.DPMSolverSinglestepScheduler,
    'dpm2': diffusers.KDPM2DiscreteScheduler,
    'dpm2 a': diffusers.KDPM2AncestralDiscreteScheduler
}

app = Flask(__name__)
pipe = None

def run_pipe(prompt, width, height, num_steps, guidance, strength, neg_prompt, img, gen, stream, pipe, cb_queue, res_queue):
    # sample callback
    def sample_cb(iter, t, latents):
        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            sample = pipe.vae.decode(latents).sample
            sample = (sample / 2 + 0.5).clamp(0, 1)

            sample = sample.cpu().permute(0, 2, 3, 1).float().numpy()
            image = pipe.numpy_to_pil(sample)

            buffer = io.BytesIO()
            image[0].save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            cb_queue.put(img_base64)

    res = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        strength=strength,
        negative_prompt=[neg_prompt],
        image=img,
        generator=gen,
        callback = sample_cb if stream else None,
        callback_steps=stream,
    ).images[0]

    cb_queue.put(None)
    res_queue.put(res)


@app.route('/generate', methods=['POST'])
def generate():
    def _generate():
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        # get prompt
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # get prameters
        model = data.get('model', DEFAULT_MODEL)
        width = data.get('width', 512)
        height = data.get('height', 512)
        num_steps = data.get('num_steps', 25)
        guidance = data.get('guidance', 7.0)
        sampler = data.get('sampler', 'dpm++ 2m')
        seed = data.get('seed', random.randint(0, 2**64 - 1))
        neg_prompt = data.get('neg', '')
        stream = data.get('stream', None)
        img = data.get('img', None)
        strength = data.get('strength', 0.8)

        # check sampler
        if sampler not in SAMPLERS:
            return jsonify({"error": f"Invalid sampler '{sampler}'. Available samplers: {list(SAMPLERS.keys())}"}), 400

        # load model
        pipe = None
        input_img = None

        if img:
            # img2img
            pipe = diffusers.StableDiffusionImg2ImgPipeline.from_single_file(model, torch_dtype=fp_prec)
            img_data = base64.b64decode(img)
            input_img = Image.open(io.BytesIO(img_data))
        else:
            # txt2img
            pipe = diffusers.StableDiffusionPipeline.from_single_file(model, torch_dtype=fp_prec)

        pipe.unet.set_attn_processor(diffusers.models.attention_processor.AttnProcessor2_0())
        pipe.scheduler = SAMPLERS[sampler].from_config(pipe.scheduler.config)
        pipe.safety_checker = lambda images, clip_input: (images, None)

        pipe.to(dev, fp_prec)

        # init generator
        gen = torch.Generator(dev).manual_seed(seed)

        try:
            data = {
                'prompt': prompt,
                'neg_prompt': neg_prompt,
                'seed': seed,
                'sampler': sampler,
                'w': width,
                'h': height,
                'num_steps': num_steps,
                'guidance': guidance,
                'stream': stream,
                'img': img,
                'strength': strength
            }
            print(f'Generating image: {json.dumps(data)}')

            cb_queue = queue.Queue()
            res_queue = queue.Queue()

            proc = threading.Thread(target=run_pipe, args=(prompt, width, height, num_steps, guidance, strength, neg_prompt, input_img, gen, stream, pipe, cb_queue, res_queue))
            proc.start()

            # send samples
            while True:
                img = cb_queue.get()
                if img is None:
                    break
                yield f'{json.dumps({"img": img, "seed": seed})}\n'

            # wait final output
            image = res_queue.get()

            del pipe
            torch.cuda.empty_cache()

            # image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # answer
            print("Image generated and encoded successfully.")
            yield f'{json.dumps({"img": img_base64, "seed": seed})}\n'

        except Exception as e:
            print(f"Error during image generation: {e}")
            return jsonify({"error": "Internal server error during image generation", "details": str(e)}), 500
    return Response(stream_with_context(_generate()), mimetype='text/json')


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SD image generator server', add_help=False)
    parser.add_argument('--host', default='0.0.0.0', type=str, help='Server host address')
    parser.add_argument('-p', '--port', default=5000, type=int, help='Server port')
    parser.add_argument('-d', '--device', default=DEFAULT_DEVICE, type=str,  choices=['cpu', 'cuda', 'mps'], help='Model compute device')
    parser.add_argument('-f', '--fp_prec', default=32, type=int, choices=[16, 32], help='Model floating point precision')
    args = parser.parse_args()

    dev = args.device
    fp_prec = {16: torch.float16, 32: torch.float32}[args.fp_prec]

    app.run(host=args.host, port=args.port, debug=False)
