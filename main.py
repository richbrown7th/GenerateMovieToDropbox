import os, tempfile, subprocess, argparse, dropbox, torch, cv2
from tqdm import tqdm
from diffusers import StableVideoDiffusionPipeline
from config import *

def generate_video(prompt, tmpdir):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        SD_MODEL, torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_model_cpu_offload()
    num_frames = FPS * DURATION
    print(f"[INFO] Generating {num_frames} frames...")
    output = pipe(prompt=prompt, num_inference_steps=20, num_frames=num_frames)
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

def upload_to_dropbox(path):
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    with open(path, "rb") as f:
        dbx.files_upload(f.read(), f"/AI_Videos/{os.path.basename(path)}", mode=dropbox.files.WriteMode("overwrite"))
    print("[INFO] Uploaded to Dropbox:", os.path.basename(path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        low = generate_video(args.prompt, tmp)
        up = upscale_with_topaz(low, tmp)
        upload_to_dropbox(up)

if __name__ == "__main__":
    main()
