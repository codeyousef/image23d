# explore_model.py
from huggingface_hub import list_repo_files, hf_hub_url


def explore_hunyuan_model(repo_id):
    print(f"\nExploring {repo_id}:")
    print("=" * 50)

    try:
        files = list_repo_files(repo_id)

        # Group files by type
        configs = []
        models = []
        others = []

        for file in files:
            if file.endswith(('.json', '.yaml', '.yml')):
                configs.append(file)
            elif file.endswith(('.pth', '.pt', '.safetensors', '.bin', '.ckpt')):
                models.append(file)
            else:
                others.append(file)

        print(f"\nConfig files:")
        for f in configs[:10]:  # Show first 10
            print(f"  - {f}")

        print(f"\nModel files:")
        for f in models[:10]:
            print(f"  - {f}")

        print(f"\nOther files:")
        for f in others[:10]:
            print(f"  - {f}")

        print(f"\nTotal files: {len(files)}")

    except Exception as e:
        print(f"Error: {e}")


# Check all versions
for model in ["tencent/Hunyuan3D-2.1", "tencent/Hunyuan3D-2", "tencent/Hunyuan3D-2mini"]:
    explore_hunyuan_model(model)