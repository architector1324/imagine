#!/usr/bin/python

import io
import json
import base64
import random
import datetime
import argparse
import requests

from PIL import Image


IMAGINE_URL = 'http://localhost:5000/generate'
DEFAULT_MODEL = '/home/arch/AI/models/dreamshaper_8.safetensors'
SAMPLERS = ['ddim', 'euler', 'euler a', 'heun', 'lms', 'dpm++ 2m', 'dpm++ 2s', 'dpm++ sde', 'dpm2', 'dpm2 a']

# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SD image generator', add_help=False)

    parser.add_argument('-m', '--model', default=DEFAULT_MODEL, type=str, help='SD model')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output image')
    parser.add_argument('-w', '--width', default=512, type=int, help='Output image width')
    parser.add_argument('-h', '--height', default=512, type=int, help='Output image height')
    parser.add_argument('-n', '--num_steps', default=25, type=int, help='Number of steps')
    parser.add_argument('-g', '--guidance', default=7.0, type=float, help='Guidance scale')
    parser.add_argument('-d', '--strength', default=0.8, type=float, help='Denoising strength (only if `--img` provided)')
    parser.add_argument('-s', '--sampler', default='dpm++ 2m', type=str, help=f'SD Sampler {SAMPLERS}')
    parser.add_argument('-i', '--img', default=None, type=str, help='Input image')
    parser.add_argument('-f', '--hires', default=1.5, type=float, help='High Resolution fix')
    parser.add_argument('--seed', default=random.randint(0, 2**64 - 1), type=int, help='Seed')
    parser.add_argument('--neg', default='', type=str, help='Negative prompt')
    parser.add_argument('--stream', default=None, type=int, help='Stream steps samples to output image')
    parser.add_argument('prompt', nargs='+', type=str, help='Prompt for model')
    parser.add_argument('--help', action='help')

    args = parser.parse_args()

    try:
        # img2img
        img_base64 = None
        if args.img:
            img = Image.open(args.img).convert("RGB").resize((args.width, args.height))
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

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
            'stream': args.stream,
            'img': img_base64,
            'strength': args.strength
        }

        meta = {
            'meta': payload,
            'out': ''
        }

        response = requests.post(IMAGINE_URL, json=payload, stream=args.stream)
        response.raise_for_status()

        filename = args.output if args.output else f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        meta_filename = f'{filename}.json'

        # process
        steps = 0
        for msg in response.iter_lines():
            result = json.loads(msg)
            # print(result)

            if 'img' in result:
                seed = result.get('seed')
                img_data = base64.b64decode(result['img'])
                image = Image.open(io.BytesIO(img_data))

                meta['meta']['seed'] = seed
                meta['out'] = result['img']

                image.save(filename)
                with open(meta_filename, 'w') as f:
                    f.write(json.dumps(meta, indent=2, ensure_ascii=False))

                print(f'Image saved [{steps}/{args.num_steps}]: {filename}')
                steps += 1
            elif 'error' in result:
                print(f'Server error: {result['error']}')
                if 'details' in result:
                    print(f'Details: {result['details']}')
        
        # high resolution fix
        if args.hires and not args.img:
            w = int(args.width * args.hires)
            h = int(args.height * args.hires)

            img_data = base64.b64decode(result['img'])
            img = Image.open(io.BytesIO(img_data)).resize((w, h))

            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            payload = {
                'model': args.model,
                'prompt': args.prompt,
                'width': w,
                'height': h,
                'num_steps': args.num_steps,
                'guidance': args.guidance,
                'sampler': args.sampler,
                'seed': args.seed,
                'neg': args.neg,
                'stream': args.stream,
                'img': img_base64,
                'strength': 0.35
            }

            response = requests.post(IMAGINE_URL, json=payload, stream=args.stream)
            response.raise_for_status()

            steps = 0
            for msg in response.iter_lines():
                result = json.loads(msg)
                # print(result)

                if 'img' in result:
                    seed = result.get('seed')
                    img_data = base64.b64decode(result['img'])
                    image = Image.open(io.BytesIO(img_data)).resize((args.width, args.height))

                    meta['meta']['seed'] = seed
                    meta['out'] = result['img']

                    image.save(filename)
                    with open(meta_filename, 'w') as f:
                        f.write(json.dumps(meta, indent=2, ensure_ascii=False))

                    print(f'Image hires.fix saved [{steps}/{args.num_steps}]: {filename}')
                    steps += 1
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
