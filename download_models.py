from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "liuhaotian/llava-v1.5-7b"
FILENAME = "mm_projector.bin"

model = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
)