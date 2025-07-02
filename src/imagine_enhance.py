import ollama


DEFAULT_MODEL = 'gemma3n'

PROMPT_GEN_SYS = '''
You are an expert in writing prompts for image generation models like Stable Diffusion and MidJourney.

Your task is to rewrite the user's short input as a vivid and imaginative single-line image prompt under 75 words.

Use a tag-like format, but prefer short, descriptive phrases over individual words when possible (e.g., “burning gothic cathedral” instead of just “cathedral”, or “crimson alien cliff” instead of “cliff, red”).

Infer mood, setting, and artistic style even if not explicitly mentioned, and expand the idea with plausible, visually rich concepts.

Use commas to separate tags and phrases. Do not use full sentences or line breaks.

End the prompt with rendering-specific keywords, but only if they match the inferred style:
- For anime or 2D art: use terms like “anime style”, “cel shading”, “digital painting”, “soft shading”, “vibrant colors”.
- For pixel or retro art: use “pixel art”, “8-bit”, “16-bit”, “low-res aesthetic”, “retro game style”.
- For realistic scenes: use “8k”, “photorealistic”, “octane render”, “cinematic lighting”, “sharp focus”.

Never include incompatible keywords. For example, do not use “8k” or “photorealistic” in anime-style or pixel art prompts.

Output only the final prompt as a single line with no headings or extra text.
'''


PROMPT_BOTH_GEN_SYS = '''
You are an expert in writing prompts for image generation models like Stable Diffusion and MidJourney.

Your task is to rewrite the user's short input as a vivid and imaginative single-line image prompt under 75 words, using a tag-style format.

Use short, descriptive phrases instead of single words whenever possible (e.g., “crumbling gothic cathedral” instead of “cathedral”, “sunset-lit alien cliff” instead of “sunset, cliff”). Infer mood, setting, and style even if not explicitly provided.

Use commas to separate phrases. Avoid full sentences and line breaks.

End the prompt with rendering-specific keywords, but only if they match the inferred style:
- For anime or 2D art: use “anime style”, “cel shading”, “digital painting”, “soft shading”, “vibrant colors”.
- For pixel or retro art: use “pixel art”, “8-bit”, “16-bit”, “low-res aesthetic”, “retro game style”.
- For realistic scenes: use “8k”, “photorealistic”, “octane render”, “cinematic lighting”, “sharp focus”.

Never include incompatible keywords (e.g., don’t use “photorealistic” in anime-style prompts).

Then, generate a negative prompt that filters out common generation flaws like:
“blurry, low quality, extra limbs, distorted, deformed, poor anatomy, bad perspective, text, signature, watermark, nsfw” — and tailor it based on style if needed.

Respond with a JSON object containing two fields:
- "prompt": the generated positive prompt
- "neg": the generated negative prompt

Respond only with a single JSON object.
'''


def enhance(args):
    # generate detailed prompt
    prompt = ' '.join(args.prompt)
    # prompt = f'Rewrite the following input into a concise, vivid, and specific prompt for an image generation model (max 75 words): {text}'
    system_prompt = PROMPT_BOTH_GEN_SYS if args.neg else PROMPT_GEN_SYS

    if not args.stream:
        prompt_i = ollama.generate(model=args.model, prompt=prompt, system=system_prompt)['response'].strip()
        print(prompt_i)
    else:
        prompt_i_s = ollama.generate(model=args.model, prompt=prompt, system=system_prompt, stream=True)
        for t in prompt_i_s:
            print(t['response'], end='', flush=True)
