# GenerateMovieToDropbox

Generates a short video from a prompt using Stable Diffusion, upscales it with Topaz Video AI, and uploads to Dropbox.

## Quickstart

1. Install Python dependencies
2. Set your Dropbox token and Topaz CLI path in `config.py` or via GitHub Secrets
3. Run:
   ```bash
   python main.py --prompt "A futuristic skyline"
   ```

## GitHub Actions

You can trigger the whole pipeline from the GitHub UI via the included workflow file.
