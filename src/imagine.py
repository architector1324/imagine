#!/usr/bin/env python

import random
import argparse

import imagine_run
import imagine_list
import imagine_enhance
import imagine_server_defs


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SD image generator CLI', add_help=False)
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    parser.add_argument('--help', action='help')

    # run
    run_parser = subparsers.add_parser('run', help='SD image generator', add_help=False)
    run_parser.add_argument('-m', '--model', default=imagine_run.DEFAULT_MODEL, type=str, help='SD model')
    run_parser.add_argument('-o', '--output', default=None, type=str, help='Output image')
    run_parser.add_argument('--meta', action='store_true', help='Save meta json of output image')
    run_parser.add_argument('-w', '--width', default=512, type=int, help='Output image width')
    run_parser.add_argument('-h', '--height', default=512, type=int, help='Output image height')
    run_parser.add_argument('-n', '--steps', default=25, type=int, help='Number of steps')
    run_parser.add_argument('-g', '--guidance', default=7.0, type=float, help='Guidance scale')
    run_parser.add_argument('-d', '--strength', default=0.8, type=float, help='Denoising strength (only if `--img` provided)')
    run_parser.add_argument('-c', '--clip', default=1, type=int, help='Clip skip')
    run_parser.add_argument('--sampler', default='dpm++ 2m', type=str, help=f'SD Sampler {imagine_run.SAMPLERS}')
    run_parser.add_argument('-i', '--img', default=None, type=str, help='Input image')
    run_parser.add_argument('-f', '--hires', default=None, type=float, help='High Resolution fix')
    run_parser.add_argument('--seed', default=random.randint(0, 2**64 - 1), type=int, help='Seed')
    run_parser.add_argument('--neg', default='ugly, deformed, blurry, low quality', type=str, help='Negative prompt')
    run_parser.add_argument('-s', '--stream', default=None, type=int, help='Stream steps samples to output image')
    run_parser.add_argument('prompt', nargs='+', type=str, help='Prompt for model')
    run_parser.add_argument('-a', '--address', default='0.0.0.0:5000', type=str, help='Server host address')
    run_parser.add_argument('--help', action='help')

    # info
    info_parser = subparsers.add_parser('info', help='Get generated image meta information', add_help=False)
    info_parser.add_argument('img',  type=str, help='Input image')
    info_parser.add_argument('--help', action='help')

    # convert
    convert_parser = subparsers.add_parser('convert', help='Convert generated image to json meta and vice versa', add_help=False)
    convert_parser.add_argument('filename',  type=str, help='JSON metadata or generated image')
    convert_parser.add_argument('--help', action='help')

    # enhance
    enhance_parser = subparsers.add_parser('enhance', help='Prompt enhancer with LLM', add_help=False)
    enhance_parser.add_argument('-m', '--model', default=imagine_enhance.DEFAULT_MODEL, type=str, help='LLM model to use')
    enhance_parser.add_argument('-s', '--stream', action='store_true', default=False, help='Stream output')
    enhance_parser.add_argument('-n', '--neg', action='store_true', default=False, help='Include negative prompt')
    enhance_parser.add_argument('-a', '--address', default='0.0.0.0:11434', type=str, help='Ollama server host address')
    enhance_parser.add_argument('prompt', nargs='+', type=str, help='Simplified prompt')
    enhance_parser.add_argument('--help', action='help')

    # server
    server_parser = subparsers.add_parser('serve', help='SD image generator server', add_help=False)
    server_parser.add_argument('--host', default='0.0.0.0', type=str, help='Server host address')
    server_parser.add_argument('-m', '--models', default=imagine_server_defs.DEFAULT_MODELS_PATH, type=str, help='SD models path')
    server_parser.add_argument('-p', '--port', default=5000, type=int, help='Server port')
    server_parser.add_argument('-d', '--device', default=imagine_server_defs.DEFAULT_DEVICE, type=str,  choices=['cpu', 'cuda', 'mps'], help='Model compute device')
    server_parser.add_argument('-f', '--full_prec', action='store_true', help='Use full (float32) floating point precision instead of float16 (default).')
    server_parser.add_argument('--help', action='help')

    # list
    list_parser = subparsers.add_parser('list', help='List available models', add_help=False)
    list_parser.add_argument('-a', '--address', default='0.0.0.0:5000', type=str, help='Server host address')
    list_parser.add_argument('--help', action='help')

    args = parser.parse_args()

    if args.command == 'serve':
        import imagine_server
        imagine_server.serve(args)
    elif args.command == 'run':
        imagine_run.run(args)
    elif args.command == 'info':
        imagine_run.info(args)
    elif args.command == 'enhance':
        imagine_enhance.enhance(args)
    elif args.command == 'convert':
        imagine_run.convert(args)
    elif args.command == 'list':
        imagine_list.list_models(args)
    else:
        parser.print_help()
