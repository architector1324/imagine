#!/bin/python

import io
import base64
import random
import datetime
import argparse
import requests

from PIL import Image


IMAGINE_URL = 'http://localhost:5000/generate'
DEFAULT_MODEL = '/home/arch/AI/models/dreamshaper_8.safetensors'
SAMPLERS = ['DDIM', 'Euler', 'Euler a', 'Heun', 'LMS', 'DPM++ 2M', 'DPM++ 2S', 'DPM++ SDE', 'DPM2', 'DPM2 a']

# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SD image generator', add_help=False)

    parser.add_argument('-m', '--model', default=DEFAULT_MODEL, type=str, help='SD model')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output image')
    parser.add_argument('-w', '--width', default=512, type=int, help='Output image width')
    parser.add_argument('-h', '--height', default=512, type=int, help='Output image height')
    parser.add_argument('-n', '--num_steps', default=25, type=int, help='Number of steps')
    parser.add_argument('-g', '--guidance', default=7.0, type=float, help='Guidance scale')
    # parser.add_argument('-d', '--denoise_strength', default=0.8, type=float, help='Denoising strength (only if `--img` provided)')
    parser.add_argument('-s', '--sampler', default='DPM++ 2M', type=str, help=f'SD Sampler {SAMPLERS}')
    # parser.add_argument('-i', '--img', default=None, type=str, help='Input image')
    parser.add_argument('--seed', default=random.randint(0, 2**64 - 1), type=int, help='Seed')
    parser.add_argument('--neg', default='', type=str, help='Negative prompt')
    # parser.add_argument('--stream', default=None, type=int, help='Stream steps samples to output image')
    parser.add_argument('prompt', nargs='+', type=str, help='Prompt for model')
    parser.add_argument('--help', action='help')

    args = parser.parse_args()

    try:
        payload = {
            'model': args.model,
            'prompt': args.prompt,
            'width': args.width,
            'height': args.height,
            'num_steps': args.num_steps,
            'guidance': args.guidance,
            'sampler': args.sampler,
            'seed': args.seed,
            'neg': args.neg,
        }

        response = requests.post(IMAGINE_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        if 'img' in result:
            img_data = base64.b64decode(result['img'])
            image = Image.open(io.BytesIO(img_data))
            filename = args.output if args.output else f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

            image.save(filename)
            print(f'Image saved as {filename} with seed: {result.get('seed')}')
        elif 'error' in result:
            print(f'Server error: {result['error']}')
            if 'details' in result:
                print(f'Details: {result['details']}')
    
    except requests.exceptions.ConnectionError as e:
        print(f'Could not connect to the server. Is it running? Error: {e}')
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
