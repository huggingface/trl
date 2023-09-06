from huggingface_hub import HfApi


api = HfApi()

api.upload_folder(
    folder_path="benchmark/trl",
    path_in_repo="images/benchmark",
    repo_id="trl-internal-testing/example-images",
    repo_type="dataset",
)
