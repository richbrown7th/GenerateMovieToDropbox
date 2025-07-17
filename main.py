import os, tempfile, subprocess, argparse, dropbox, torch, cv2
from tqdm import tqdm
from diffusers import StableVideoDiffusionPipeline
from config import *

def generate_video(prompt, tmpdir):
    
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        SD_MODEL, torch_dtype=dtype
    ).to(device)
    
    #pipe.enable_model_cpu_offload()
    num_frames = FPS * DURATION
    print(f"[INFO] Generating {num_frames} frames...")
    output = pipe(text=prompt, num_inference_steps=20, num_frames=num_frames)
    low_vid = os.path.join(tmpdir, "lowres.mp4")
    writer = cv2.VideoWriter(low_vid, cv2.VideoWriter_fourcc(*"mp4v"), FPS, RES_LOW)
    for frame in output.frames:
        img = cv2.cvtColor(frame.numpy(), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, RES_LOW)
        writer.write(img)
    writer.release()
    return low_vid

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
