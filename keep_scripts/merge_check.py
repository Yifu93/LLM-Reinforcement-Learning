import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def check_model_directory(model_path: str):
    """
    Verifies that `model_path` exists, contains the necessary files,
    and can be loaded by Hugging Face Transformers.
    """
    print(f"Checking model path: {model_path!r}")
    # 1. Check that the directory exists
    if not os.path.isdir(model_path):
        print("  ✗ Directory does not exist.")
        return False

    # 2. List files in the directory
    files = os.listdir(model_path)
    print("  • Files in directory:", files)

    # 3. Check for a config.json
    if "config.json" not in files:
        print("  ✗ Missing 'config.json'.")
        return False
    else:
        print("  ✓ Found 'config.json'.")

    # 4. Try to load the config
    try:
        _ = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print("  ✓ AutoConfig.from_pretrained succeeded.")
    except Exception as e:
        print("  ✗ AutoConfig.from_pretrained failed:")
        print("    ", e)
        return False

    # 5. Try to load the tokenizer (if present)
    #    This checks for tokenizer.json, tokenizer_config.json, or related files
    try:
        _ = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("  ✓ AutoTokenizer.from_pretrained succeeded.")
    except Exception as e:
        print("  ✗ AutoTokenizer.from_pretrained failed:")
        print("    ", e)
        # We continue, since some models may not provide a tokenizer here
        # but usually causal-LMs do.

    # 6. Try to load the model itself
    try:
        _ = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        print("  ✓ AutoModelForCausalLM.from_pretrained succeeded.")
    except Exception as e:
        print("  ✗ AutoModelForCausalLM.from_pretrained failed:")
        print("    ", e)
        return False

    print("All checks passed. Model directory is valid.")
    return True


if __name__ == "__main__":
    # Replace this path with the one you want to validate
    model_dir = "/checkpoints/merged_DPO_ultrafeedback_lora"

    success = check_model_directory(model_dir)
    if not success:
        print("\nModel load check failed. Please verify:\n"
              "1. `model_dir` is a correct local path to your model folder.\n"
              "2. The folder contains a valid `config.json` (and optionally `pytorch_model.bin`).\n"
              "3. If you intended a Hugging Face Hub repo, use its `namespace/repo_name` instead of a local path.")
    else:
        print("\nModel load check succeeded. You can now pass this path to vLLM or Transformers.")
