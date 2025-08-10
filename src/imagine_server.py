import os
import io
import json
import queue
import torch
import base64
import random
import threading
import diffusers

import imagine_server_defs

from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver

from PIL import Image

# settings
DEFAULT_FP_PREC = torch.float16

os.environ['HF_HUB_OFFLINE'] = '1'

fp_prec = DEFAULT_FP_PREC
dev = imagine_server_defs.DEFAULT_DEVICE
models_path = imagine_server_defs.DEFAULT_MODELS_PATH

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

def run_pipe(prompt, width, height, steps, guidance, strength, clip_skip, neg_prompt, img, gen, stream, pipe, cb_queue, res_queue, stop_event):
    # sample callback
    def sample_cb(iter, t, latents):
        # Check if the stop_event is set; if so, signal diffusers to stop
        if stop_event.is_set():
            print(f"Generation (seed {gen.initial_seed()}) cancelled due to client disconnect signal.")
            raise Exception('Generation was cancelled by client.')

        with torch.no_grad():
            sample = pipe.decode_latents(latents)
            image = pipe.numpy_to_pil(sample)

            buffer = io.BytesIO()
            image[0].save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            cb_queue.put(img_base64)

    try:
        res = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
            clip_skip=max(clip_skip - 1, 0),
            negative_prompt=neg_prompt,
            image=img,
            generator=gen,
            callback = sample_cb if stream else None,
            callback_steps=stream,
        ).images[0]

        # Only put the result if generation was NOT cancelled by the stop_event
        if not stop_event.is_set():
            res_queue.put(res)
        else:
            # If cancelled, put a specific signal or exception to distinguish from actual errors
            res_queue.put(Exception("Generation cancelled by client."))

    except Exception as e:
        # Catch any errors during generation and pass them to the main thread
        res_queue.put(e)
    finally:
        # Signal that callback stream is finished, or that the process is ending
        cb_queue.put(None)

        # Cleanup model from memory
        pipe.maybe_free_model_hooks()

        print(f'Pipe cleared for seed {gen.initial_seed()}.')

        if dev == 'cuda':
            torch.cuda.empty_cache()
            print(f"CUDA cache cleared for seed {gen.initial_seed()}.")


def generate_image_logic(data):
    # get prompt
    prompt = data.get('prompt')
    if not prompt:
        raise ValueError("Prompt is required")

    # get model
    model_name = data.get('model', None)
    if not model_name:
        raise ValueError('Models is required')

    model_path = os.path.join(os.path.expanduser(models_path), f'{model_name}.safetensors')

    # get parameters
    width = data.get('width', 512)
    height = data.get('height', 512)
    steps = data.get('steps', 25)
    guidance = data.get('guidance', 7.0)
    sampler = data.get('sampler', 'dpm++ 2m')
    seed_str = data.get('seed', str(random.randint(0, 2**64 - 1)))
    neg_prompt = data.get('neg', '')
    stream = data.get('stream', None)
    img_b64 = data.get('img', None)
    strength = data.get('strength', 0.8)
    clip_skip = data.get('clip', 1)

    seed = int(seed_str)

    # check sampler
    if sampler not in SAMPLERS:
        raise ValueError(f"Invalid sampler '{sampler}'. Available samplers: {list(SAMPLERS.keys())}")

    # load model
    pipe = None
    input_img = None

    print(f"Loading model: {model_path} for device {dev}, precision {fp_prec}")
    if img_b64:
        # img2img
        pipe = diffusers.StableDiffusionImg2ImgPipeline.from_single_file(model_path, torch_dtype=fp_prec)
        img_data = base64.b64decode(img_b64)
        input_img = Image.open(io.BytesIO(img_data)).convert("RGB").resize((width, height))
    else:
        # txt2img
        pipe = diffusers.StableDiffusionPipeline.from_single_file(model_path, torch_dtype=fp_prec)

    pipe.unet.set_attn_processor(diffusers.models.attention_processor.AttnProcessor2_0())
    pipe.scheduler = SAMPLERS[sampler].from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, clip_input: (images, None) # Disable safety checker
    pipe.to(dev, fp_prec)

    pipe.unet.to(fp_prec)
    pipe.vae.to(fp_prec)
    pipe.text_encoder.to(fp_prec)

    # init generator
    gen = torch.Generator(dev).manual_seed(seed)

    # Setup communication for generation thread
    cb_queue = queue.Queue() # Queue for intermediate callbacks from run_pipe
    res_queue = queue.Queue() # Queue for final result or error from run_pipe
    stop_event = threading.Event() # Event to signal run_pipe to stop early

    # Start generation in a separate thread
    proc = threading.Thread(target=run_pipe, args=(
        prompt, width, height, steps, guidance, strength, clip_skip, neg_prompt,
        input_img, gen, stream, pipe, cb_queue, res_queue, stop_event
    ))
    proc.start()

    try:
        request_log_data = {
            'model': model_name,
            'prompt': prompt,
            'neg': neg_prompt,
            'seed': seed,
            'sampler': sampler,
            'width': width,
            'height': height,
            'steps': steps,
            'guidance': guidance,
            'stream': stream,
            'img': f'{img_b64[:32]}...' if img_b64 else None,
            'strength': strength,
            'clip': clip_skip
        }
        print(f'Generating image: {json.dumps(request_log_data)}')

        # Send samples if streaming is enabled
        if stream:
            while True:
                # Use a small timeout to allow the generator to be closed from outside,
                # ensuring GeneratorExit can be handled in the outer try block.
                try:
                    img = cb_queue.get(timeout=0.1) # Blocks for max 0.1s
                    if img is None: # Signal from run_pipe that no more callbacks will come
                        break
                    # Yield JSON string with newline for stream processing
                    yield json.dumps({"img": img, "seed": str(seed), "status": "intermediate"}) + '\n'
                except queue.Empty:
                    # If queue is empty, check if the worker thread is still alive and not just waiting for data.
                    # This helps prevent infinite loops if the worker died without signaling None.
                    if not proc.is_alive() and cb_queue.empty() and res_queue.empty():
                        print(f"Warning: Generation thread for seed {seed} ended prematurely or without final signal.")
                        break # Exit loop if worker thread is gone and no more data expected
                    continue # Keep trying to get from queue

        # Wait for final output (only if generation was not cancelled externally via GeneratorExit)
        # This part will be reached if streaming finishes naturally or if not streaming.
        # If GeneratorExit was raised, this block won't be entered directly from `yield` loop.
        # Check stop_event here, in case client disconnected without streaming, or at the very end.
        if not stop_event.is_set():
            final_result = res_queue.get() # Blocking wait for final result
            if isinstance(final_result, Exception):
                raise final_result # Re-raise error from run_pipe or cancellation signal

            final_image = final_result

            # image to base64
            buffer = io.BytesIO()
            final_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            print("Image generated and encoded successfully.")
            # Yield the final result
            yield json.dumps({"img": img_base64, "seed": str(seed), "status": "final"}) + '\n'

    except GeneratorExit:
        # This exception is raised by the caller (.close() on generator) when the client disconnects.
        print(f"Generator for seed {seed} detected client disconnect. Signaling generation thread to stop.")
        stop_event.set()
    except Exception as e:
        # Catch any other exceptions during the logic of this generator
        print(f"Error in generate_image_logic for seed {seed}: {e}")
        stop_event.set() 
        raise

    finally:
        # This block is *always* executed when the generator finishes or is closed (e.g., by GeneratorExit).
        # Ensures the generation thread is joined and resources are cleaned up.
        if proc.is_alive():
            print(f"Waiting for generation thread (seed {seed}) to finish cleanup...")
            stop_event.set() # Ensure the event is set, just in case
            proc.join(timeout=5) # Give it some time to stop gracefully
            if proc.is_alive():
                print(f"Warning: Generation thread (seed {seed}) did not stop gracefully. It might still be running or stuck.")


def get_models():
    expanded_models_path = os.path.expanduser(models_path)
    os.makedirs(expanded_models_path, exist_ok=True)

    model_names = []
    # read models
    for filename in os.listdir(expanded_models_path):
        # filter
        if filename.endswith(('.safetensors', '.ckpt')):
            model_name = os.path.splitext(filename)[0]
            model_names.append(model_name)
    
    # sort
    model_names.sort()
    return model_names

class SDRequestHandler(BaseHTTPRequestHandler):
    # Handle OPTIONS preflight requests
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS') # Allow POST and OPTIONS methods
        self.send_header('Access-Control-Allow-Headers', 'Content-Type') # Allow Content-Type header
        self.send_header('Access-Control-Max-Age', '86400') # Cache preflight response for 24 hours
        self.end_headers()

    def do_GET(self):
        if self.path == '/models':
            try:
                # read models
                model_names = get_models()

                # response
                response_data = {"models": model_names}
                response_body = json.dumps(response_data).encode('utf-8')

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
                print(f"Served /models request. Found {len(model_names)} models.")

            except Exception as e:
                print(f"Error serving /models request: {e}")
                error_response = json.dumps({"error": f"Failed to list models: {e}"}).encode('utf-8')
                self.send_error(500, "Internal Server Error", error_response)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "Request body is empty")
                return

            # Initialize to None for cleanup in except/finally blocks
            response_generator = None

            try:
                # Read the request body
                post_body = self.rfile.read(content_length).decode('utf-8')
                data = json.loads(post_body)

                is_streaming_requested = data.get('stream', None) is not None

                # Generate image logic returns a generator
                response_generator = generate_image_logic(data)

                if is_streaming_requested:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                    self.send_header('Transfer-Encoding', 'chunked')
                    self.end_headers()

                    # Send chunks
                    for chunk_data_line in response_generator:
                        encoded_chunk = chunk_data_line.encode('utf-8')
                        self.wfile.write(f'{len(encoded_chunk):X}\r\n'.encode('ascii'))
                        self.wfile.write(encoded_chunk)
                        self.wfile.write(b'\r\n')
                    self.wfile.write(b'0\r\n\r\n')

                else:
                    # If not streaming, consume the generator to get the single final result
                    final_response_json_line = next(response_generator)
                    encoded_response = final_response_json_line.encode('utf-8')

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(encoded_response)))
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                    self.end_headers()
                    self.wfile.write(encoded_response)

            except BrokenPipeError:
                # Client disconnected. Log it and close the generator to signal cleanup.
                print(f"Client disconnected from {self.path} prematurely.")
                if response_generator is not None:
                    # This will raise GeneratorExit in generate_image_logic,
                    # which will then set the stop_event for run_pipe and handle cleanup.
                    response_generator.close()
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON in request body",
                                json.dumps({"error": "Invalid JSON"}))
            except ValueError as e:
                # Catch validation errors from generate_image_logic
                self.send_error(400, str(e),
                                json.dumps({"error": str(e)}))
            except Exception as e:
                # Catch any other unexpected errors during generation or processing
                print(f"Server error during generation: {e}")

                if response_generator is not None:
                    response_generator.close() # Best effort cleanup
                self.send_error(500, "Internal Server Error",
                                json.dumps({"error": "Internal server error during image generation", "details": str(e)}))
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    # Allow threads to exit when the main program exits
    daemon_threads = True


def serve(args):
    global dev
    global fp_prec
    global models_path

    dev = args.device
    fp_prec = torch.float32 if args.full_prec else (torch.bfloat16 if args.device == 'cpu' else torch.float16)
    models_path = args.models

    server_address = (args.host, args.port)
    httpd = ThreadedHTTPServer(server_address, SDRequestHandler)
    print(f'Starting server on http://{args.host}:{args.port}')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nServer is shutting down.')
        httpd.shutdown()
        httpd.server_close()
