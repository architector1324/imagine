#!/usr/bin/env python

import random
import argparse

import imagine_run
import imagine_server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SD image generator CLI', add_help=False)
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    parser.add_argument('--help', action='help')

    # run
    run_parser = subparsers.add_parser('run', help='SD image generator', add_help=False)
    run_parser.add_argument('-m', '--model', default=imagine_run.DEFAULT_MODEL, type=str, help='SD model')
    run_parser.add_argument('-o', '--output', default=None, type=str, help='Output image')
    run_parser.add_argument('-w', '--width', default=512, type=int, help='Output image width')
    run_parser.add_argument('-h', '--height', default=512, type=int, help='Output image height')
    run_parser.add_argument('-n', '--num_steps', default=25, type=int, help='Number of steps')
    run_parser.add_argument('-g', '--guidance', default=7.0, type=float, help='Guidance scale')
    run_parser.add_argument('-d', '--strength', default=0.8, type=float, help='Denoising strength (only if `--img` provided)')
    run_parser.add_argument('--sampler', default='dpm++ 2m', type=str, help=f'SD Sampler {imagine_run.SAMPLERS}')
    run_parser.add_argument('-i', '--img', default=None, type=str, help='Input image')
    run_parser.add_argument('-f', '--hires', default=None, type=float, help='High Resolution fix')
    run_parser.add_argument('--seed', default=random.randint(0, 2**64 - 1), type=int, help='Seed')
    run_parser.add_argument('--neg', default='ugly, deformed, blurry, low quality', type=str, help='Negative prompt')
    run_parser.add_argument('-s', '--stream', default=None, type=int, help='Stream steps samples to output image')
    run_parser.add_argument('prompt', nargs='+', type=str, help='Prompt for model')
    run_parser.add_argument('-a', '--address', default='0.0.0.0:5000', type=str, help='Server host address')
    run_parser.add_argument('--help', action='help')

    # server
    server_parser = subparsers.add_parser('serve', help='SD image generator server', add_help=False)
    server_parser.add_argument('--host', default='0.0.0.0', type=str, help='Server host address')
    server_parser.add_argument('-p', '--port', default=5000, type=int, help='Server port')
    server_parser.add_argument('-d', '--device', default=imagine_server.DEFAULT_DEVICE, type=str,  choices=['cpu', 'cuda', 'mps'], help='Model compute device')
    server_parser.add_argument('-f', '--full_prec', action='store_true', help='Use full (float32) floating point precision instead of float16 (default).')
    server_parser.add_argument('--help', action='help')

    args = parser.parse_args()

    if args.command == 'serve':
        imagine_server.serve(args)
    elif args.command == 'run':
        imagine_run.run(args)
    else:
        parser.print_help()
