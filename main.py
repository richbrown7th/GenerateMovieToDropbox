import os, tempfile, subprocess, argparse, dropbox
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from config import *

# Load pipelines once
text2img_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
text2img_pipe.to("cuda" if torch.cuda.is_available() else "cpu")

video_pipe = StableVideoDiffusionPipeline.from_pretrained(SVD_MODEL_PATH, torch_dtype=torch.float16)
video_pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_video(prompt, output_dir):
    print(f"[INFO] Generating image from prompt: {prompt}")

    # Step 1: Prompt → Image
    image = text2img_pipe(prompt=prompt, guidance_scale=7.5, num_inference_steps=30).images[0]
    image_path = os.path.join(output_dir, "input_image.png")
    image.save(image_path)

    print("[INFO] Generating 32 frames of video from image...")

    # Step 2: Image → Video
    output = video_pipe(
        image=image,
        num_frames=32,
        num_inference_steps=25,
        generator=torch.manual_seed(42)
    )

    video_path = os.path.join(output_dir, "output.mp4")
    output.frames[0].save(video_path, save_all=True, append_images=output.frames[1:], duration=50, loop=0)
    return video_path

def upscale_with_topaz(input_vid, tmpdir):
    high_vid = os.path.join(tmpdir, "4k_upscaled.mp4")
    cmd = [
        TOPAZ_CLI,
        "-i", input_vid,
        "-o", high_vid,
        "-m", TOPAZ_MODEL,
        "-s", f"{RES_HIGH[0]}x{RES_HIGH[1]}",
        "--recover_details"
    ]
    subprocess.run(cmd, check=True)
    return high_vid

def upload_to_dropbox(path, folder):
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    target = f"/AI_Videos/{folder}/{os.path.basename(path)}"
    with open(path, "rb") as f:
        dbx.files_upload(f.read(), target, mode=dropbox.files.WriteMode("overwrite"))
    print(f"[INFO] Uploaded to Dropbox: {target}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--use_topaz", type=str, default="true", help="Enable or disable Topaz AI (true/false)")
    args = parser.parse_args()

    use_topaz_flag = args.use_topaz.lower() == "true"

    with tempfile.TemporaryDirectory() as tmp:
        low = generate_video(args.prompt, tmp)
        upload_to_dropbox(low, "lowres")

        if use_topaz_flag:
            up = upscale_with_topaz(low, tmp)
            upload_to_dropbox(up, "4k")

if __name__ == "__main__":
    main()