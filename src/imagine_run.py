import io
import json
import base64
import random
import datetime
import argparse
import requests

from PIL import Image


IMAGINE_URL = 'http://{address}/generate'
DEFAULT_MODEL = 'dreamshaper_8'
SAMPLERS = ['ddim', 'euler', 'euler a', 'heun', 'lms', 'dpm++ 2m', 'dpm++ 2s', 'dpm++ sde', 'dpm2', 'dpm2 a']

def send_generate_request(payload, address, stream, filename, meta, prefix="Image saved", resize=None):
    meta_filename = f'{filename}.json'

    response = requests.post(IMAGINE_URL.format(address=address), json=payload, stream=stream)
    response.raise_for_status()

    steps = 0
    for msg in response.iter_lines():
        result = json.loads(msg)

        if 'img' in result:
            seed = result.get('seed')
            img_data = base64.b64decode(result['img'])
            image = Image.open(io.BytesIO(img_data))

            # For hires.fix, we need to resize the final image back to original dimensions
            if resize is not None:
                image = image.resize(resize)

            meta['meta']['seed'] = seed
            meta['out'] = result['img']

            image.save(filename)
            with open(meta_filename, 'w') as f:
                f.write(json.dumps(meta, indent=2, ensure_ascii=False))

            print(f'{prefix} [{steps}/{payload["num_steps"]}]: {filename}')
            steps += 1
        elif 'error' in result:
            print(f'Server error: {result["error"]}')
            if 'details' in result:
                print(f'Details: {result["details"]}')
    return result


def run(args):
    prompt = ' '.join(args.prompt)

    try:
        # img2img
        img_base64 = None
        if args.img:
            img = Image.open(args.img).convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        payload = {
            'model': args.model,
            'prompt': prompt,
            'width': args.width,
            'height': args.height,
            'num_steps': args.num_steps,
            'guidance': args.guidance,
            'sampler': args.sampler,
            'seed': str(args.seed),
            'neg': args.neg,
            'stream': args.stream,
            'img': img_base64,
            'strength': args.strength
        }

        meta = {
            'meta': payload,
            'out': ''
        }

        filename = args.output if args.output else f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

        result = send_generate_request(payload, args.address, args.stream, filename, meta, prefix="Image saved")

        # high resolution fix
        if args.hires and not args.img:
            w = int(args.width * args.hires)
            h = int(args.height * args.hires)

            img_data = base64.b64decode(result['img'])
            img = Image.open(io.BytesIO(img_data)).resize((w, h))

            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            hires_payload = {
                'model': args.model,
                'prompt': prompt,
                'width': w,
                'height': h,
                'num_steps': args.num_steps,
                'guidance': args.guidance,
                'sampler': args.sampler,
                'seed': str(args.seed),
                'neg': args.neg,
                'stream': args.stream,
                'img': img_base64,
                'strength': 0.35
            }

            send_generate_request(hires_payload, args.address, args.stream, filename, meta, prefix="Image hires.fix saved", resize=(args.width, args.height))

    except requests.exceptions.ConnectionError as e:
        print(f'Could not connect to the server. Is it running? Error: {e}')
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
