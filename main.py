import os, tempfile, subprocess, argparse, dropbox, torch, cv2
import torch
from tqdm import tqdm
from diffusers import StableVideoDiffusionPipeline
from config import *

def generate_video(prompt, tmpdir):
    
    from PIL import Image
import torch

def generate_video(prompt, output_dir):
    print("[INFO] Generating 32 frames...")

    # Instead of using a prompt directly, we need an input image
    # Here's a placeholder for now — you can replace this with an actual image generator
    input_image = Image.new("RGB", (512, 512), color=(255, 100, 50))  # orange placeholder

    output = pipe(
        image=input_image,
        num_frames=32,
        num_inference_steps=25,
        generator=torch.manual_seed(42)
    )

    video_path = f"{output_dir}/output.mp4"
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
