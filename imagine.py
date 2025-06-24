#!/bin/python

import os
import io
import PIL
import json
import torch
import base64
import random
import diffusers

from flask import Flask, request, jsonify

# settings
DEFAULT_MODEL = '/home/arch/AI/models/dreamshaper_8.safetensors'
DEFAULT_FP_PREC = torch.float32
DEFAULT_DEVICE = 'cpu'

# force to use rocms
# os.environ['HSA_OVERRIDE_GFX_VERSION']='11.0.0'
# os.environ['PYTORCH_ROCM_ARCH']='gfx1100'
# os.environ['HSA_ENABLE_SDMA']='0'

SAMPLERS = {
    'DDIM': diffusers.DDIMScheduler,
    'Euler': diffusers.EulerDiscreteScheduler,
    'Euler a': diffusers.EulerAncestralDiscreteScheduler,
    'Heun': diffusers.HeunDiscreteScheduler,
    'LMS': diffusers.LMSDiscreteScheduler,
    'DPM++ 2M': diffusers.DPMSolverMultistepScheduler,
    'DPM++ 2S': diffusers.DPMSolverSinglestepScheduler,
    'DPM++ SDE': diffusers.DPMSolverSinglestepScheduler,
    'DPM2': diffusers.KDPM2DiscreteScheduler,
    'DPM2 a': diffusers.KDPM2AncestralDiscreteScheduler
}

app = Flask(__name__)
pipe = None

@app.route('/generate', methods=['POST'])
def generate():
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
    sampler = data.get('sampler', 'DPM++ 2M')
    seed = data.get('seed', random.randint(0, 2**64 - 1))
    neg_prompt = data.get('neg', '')

    # check sampler
    if sampler not in SAMPLERS:
        return jsonify({"error": f"Invalid sampler '{sampler}'. Available samplers: {list(SAMPLERS.keys())}"}), 400

    # load model
    pipe = diffusers.StableDiffusionPipeline.from_single_file(model, torch_dtype=DEFAULT_FP_PREC)

    pipe.unet.set_attn_processor(diffusers.models.attention_processor.AttnProcessor2_0())
    pipe.scheduler = SAMPLERS[sampler].from_config(pipe.scheduler.config)

    pipe.to(DEFAULT_DEVICE, DEFAULT_FP_PREC)

    # init generator
    gen = torch.Generator(DEFAULT_DEVICE).manual_seed(seed)

    try:
        data = {
            'prompt': prompt,
            'neg_prompt': neg_prompt,
            'seed': seed,
            'sampler': sampler,
            'w': width,
            'h': height,
            'num_steps': num_steps,
            'guidance': guidance
        }
        print(f'Generating image: {json.dumps(data)}')
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            negative_prompt=[neg_prompt],
            generator=gen,
        ).images[0]

        del pipe
        torch.cuda.empty_cache()

        # image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        print("Image generated and encoded successfully.")
        return jsonify({
            "img": img_base64,
            "seed": seed
        })

    except Exception as e:
        print(f"Error during image generation: {e}")
        return jsonify({"error": "Internal server error during image generation", "details": str(e)}), 500


# main
if __name__ == '__main__':
    # pipe = diffusers.StableDiffusionPipeline.from_single_file(DEFAULT_MODEL, torch_dtype=DEFAULT_FP_PREC)

    # pipe.unet.set_attn_processor(diffusers.models.attention_processor.AttnProcessor2_0())
    # pipe.scheduler = SAMPLERS['DPM++ 2M'].from_config(pipe.scheduler.config)

    # pipe.to(DEFAULT_DEVICE, DEFAULT_FP_PREC)

    app.run(host='0.0.0.0', port=5000, debug=False)
