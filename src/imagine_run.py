import io
import os
import json
import base64
import datetime
import requests

from PIL import Image
from PIL.PngImagePlugin import PngInfo


IMAGINE_URL = 'http://{address}/generate'
DEFAULT_MODEL = 'dreamshaper_8'
SAMPLERS = ['ddim', 'euler', 'euler a', 'heun', 'lms', 'dpm++ 2m', 'dpm++ 2s', 'dpm++ sde', 'dpm2', 'dpm2 a']

def send_generate_request(payload, address, stream, filename, meta, save_meta, prefix="Image saved", resize=None):
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
            meta['meta']['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            meta['out'] = result['img']

            meta_json = json.dumps(meta, indent=2, ensure_ascii=False)
            meta_meta_json = json.dumps(meta['meta'], indent=2, ensure_ascii=False)

            info = PngInfo()
            info.add_text("meta", meta_meta_json)

            image.save(filename, pnginfo=info)

            if save_meta:
                with open(meta_filename, 'w') as f:
                    f.write(meta_json)

            print(f'{prefix} [{steps}/{payload["steps"]}]: {filename}')
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
            'steps': args.steps,
            'guidance': args.guidance,
            'sampler': args.sampler,
            'seed': str(args.seed),
            'neg': args.neg,
            'stream': args.stream,
            'img': img_base64,
            'strength': args.strength,
            'clip': args.clip
        }

        meta = {
            'meta': payload,
            'out': ''
        }

        filename = args.output if args.output else f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

        result = send_generate_request(payload, args.address, args.stream, filename, meta, args.meta, prefix="Image saved")

        # high resolution fix
        if args.hires and not args.img:
            meta['meta']['hires'] = args.hires

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
                'steps': args.steps,
                'guidance': args.guidance,
                'sampler': args.sampler,
                'seed': str(args.seed),
                'neg': args.neg,
                'stream': args.stream,
                'img': img_base64,
                'strength': 0.35,
                'clip': args.clip
            }

            send_generate_request(hires_payload, args.address, args.stream, filename, meta, args.meta, prefix="Image hires.fix saved", resize=(args.width, args.height))

    except requests.exceptions.ConnectionError as e:
        print(f'Could not connect to the server. Is it running? Error: {e}')
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


def info(args):
    if not args.img:
        return

    try:
        img = Image.open(args.img)

        if 'meta' not in img.info:
            return

        meta_json = img.info['meta']
        meta = json.loads(meta_json)
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


def convert(args):
    if not args.filename:
        return

    try:
        # Определяем базовое имя файла и его расширение
        base_filename, ext = os.path.splitext(args.filename)
        ext = ext.lower()

        if ext == '.json':
            output_filename = base_filename + '.png'

            # open file
            with open(args.filename, 'r') as f:
                data = json.load(f)

            img_data_base64 = data['out']
            meta_data_for_png = data['meta']

            # decode
            img_bytes = base64.b64decode(img_data_base64)
            image = Image.open(io.BytesIO(img_bytes))

            # meta
            png_info = PngInfo()
            meta_json_string = json.dumps(meta_data_for_png, indent=2, ensure_ascii=False)
            png_info.add_text("meta", meta_json_string)

            # save image
            image.save(output_filename, pnginfo=png_info)

            print(f'Image saved: {output_filename}')

        elif ext == '.png':
            output_filename = base_filename + '.json'

            # open image
            image = Image.open(args.filename)

            # extract meta
            meta_json_string_from_png = image.info.get("meta")
            meta_data_for_json = {}
            if meta_json_string_from_png:
                meta_data_for_json = json.loads(meta_json_string_from_png)
            else:
                print(f"Warning: No 'meta' data found in {args.filename}. Output JSON 'meta' field will be empty.")

            # encode image
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            img_base64_encoded = base64.b64encode(img_bytes).decode('utf-8')

            # final json
            output_json_data = {
                "meta": meta_data_for_json,
                "out": img_base64_encoded
            }

            # save
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(output_json_data, f, indent=2, ensure_ascii=False)

            print(f'JSON saved: {output_filename}')

        else:
            print(f'Unsupported file extension: {ext}. Only .json and .png are supported for {args.filename}.')

    except Exception as e:
        print(f'An unexpected error occurred: {e}')
