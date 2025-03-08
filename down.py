from huggingface_hub import snapshot_download

# This is the official repo id
repo_id = "lmsys/vicuna-13b-v1.5"

# Download the entire repo to a local folder (by default into ~/.cache/huggingface/...)
local_folder = snapshot_download(repo_id=repo_id)
print(f"Model files downloaded to: {local_folder}")

