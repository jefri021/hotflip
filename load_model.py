import gc
import torch
from typing import List, Union

def clear_memory(keep_vars: Union[List[str], None] = None, verbose: bool = True):
    """
    Clears memory while preserving specified variables.
    Still clears GPU memory for all CUDA objects, including kept variables.
    
    Args:
        keep_vars: List of variable names to preserve in memory (will still be cleared from GPU)
        verbose: Whether to print memory clearing information
    """
    if verbose:
        print("Starting memory clearing process...")
    
    # Convert keep_vars to set for faster lookups
    keep_set = set(keep_vars) if keep_vars else set()
    
    # First pass: Move kept CUDA variables to CPU
    if torch.cuda.is_available():
        for name, var in list(globals().items()):
            if name in keep_set and isinstance(var, torch.Tensor) and var.is_cuda:
                if verbose:
                    print(f"Moving kept tensor '{name}' to CPU")
                globals()[name] = var.cpu()
    
    # Clear Python garbage collector
    gc.collect()
    if verbose:
        print("Ran Python garbage collection")
    
    # Clear CUDA memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            print("Cleared CUDA cache")
            print(f"Current CUDA memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"Current CUDA memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    # Try to clear TensorFlow/Keras if available
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        if verbose:
            print("Cleared TensorFlow/Keras session")
    except ImportError:
        pass
    
    # Delete objects not in keep_vars
    for name, var in list(globals().items()):
        if not name.startswith('__') and name not in keep_set:
            if isinstance(var, (torch.Tensor, torch.nn.Module)):
                del globals()[name]
                if verbose:
                    print(f"Deleted torch object: {name}")
            elif isinstance(var, list) and var and isinstance(var[0], torch.Tensor):
                del globals()[name]
                if verbose:
                    print(f"Deleted list of torch tensors: {name}")
    
    # Final garbage collection
    gc.collect()
    
    if verbose:
        print("Memory clearing complete")


import os
import gdown

def download_file_from_google_drive(file_id, output_dir, output_filename, quiet=False):
    """
    Downloads a file from Google Drive given its file ID and saves it to the specified directory.
    
    Args:
        file_id (str): The Google Drive file ID (found in the file URL)
        output_dir (str): Directory where the file should be saved
        output_filename (str): Name of the output file
        quiet (bool): Whether to suppress gdown output (default: False)
    
    Returns:
        str: Path to the downloaded file if successful, None otherwise
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full output path
    output_file = os.path.join(output_dir, output_filename)
    
    print("Downloading the file...")
    try:
        gdown.download(id=file_id, output=output_file, quiet=quiet, fuzzy=True)
    except Exception as e:
        print(f"Download failed: {str(e)}")
        return None
    
    # Verify download
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # in MB
        print(f"Download successful! File saved to: {output_file}")
        print(f"File size: {file_size:.2f} MB")
        return output_file
    else:
        print("Download failed - file not found")
        return None
    

import os
import tarfile
from typing import List, Union

def extract_and_delete_tar_gz(file_path: str, delete_compressed: bool = True) -> bool:
    """
    Extracts a .tar.gz file and optionally deletes the compressed file.
    
    Args:
        file_path (str): Path to the .tar.gz file
        delete_compressed (bool): Whether to delete the compressed file after extraction (default: True)
    
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        print(f"Extracting: {file_path}")
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=os.path.dirname(file_path))
        
        if delete_compressed:
            os.remove(file_path)
            print(f"Deleted compressed file: {file_path}")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def process_directory(directory: str, recursive: bool = True, max_depth: Union[int, None] = None) -> int:
    """
    Processes a directory to find and extract .tar.gz files.
    
    Args:
        directory (str): Directory path to process
        recursive (bool): Whether to process subdirectories (default: True)
        max_depth (int|None): Maximum recursion depth (None for unlimited)
    
    Returns:
        int: Number of .tar.gz files processed
    """
    processed_count = 0
    current_depth = 0
    
    while True:
        found_tar_gz = False
        for root, dirs, files in os.walk(directory):
            # Calculate current depth
            rel_path = os.path.relpath(root, directory)
            current_depth = rel_path.count(os.sep) + 1 if rel_path != '.' else 0
            
            # Skip if beyond max depth
            if max_depth is not None and current_depth > max_depth:
                continue
                
            for file in files:
                if file.endswith('.tar.gz'):
                    file_path = os.path.join(root, file)
                    if extract_and_delete_tar_gz(file_path):
                        processed_count += 1
                        found_tar_gz = True
        
        # If not recursive or no more .tar.gz files found, exit
        if not recursive or not found_tar_gz:
            break
    
    return processed_count

def process_paths(paths: List[str], recursive: bool = True, max_depth: Union[int, None] = None) -> int:
    """
    Processes a list of paths (files or directories) to extract .tar.gz files.
    
    Args:
        paths (List[str]): List of file/directory paths to process
        recursive (bool): Whether to process directories recursively (default: True)
        max_depth (int|None): Maximum recursion depth for directories (None for unlimited)
    
    Returns:
        int: Total number of .tar.gz files processed
    """
    total_processed = 0
    
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist - {path}")
            continue
            
        if path.endswith('.tar.gz'):
            if extract_and_delete_tar_gz(path):
                total_processed += 1
        elif os.path.isdir(path):
            print(f"Processing directory: {path}")
            total_processed += process_directory(
                directory=path,
                recursive=recursive,
                max_depth=max_depth
            )
    
    print(f"Total .tar.gz files processed: {total_processed}")
    return total_processed

from os.path import join

import torch
import json
import os
import logging
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_model(model_filepath: str, torch_dtype:torch.dtype=torch.float16):
    """Load a model given a specific model_path.

    Args:
        model_filepath: str - Path to where the model is stored

    Returns:
        model, dict, str - Torch model + dictionary representation of the model + model class name
    """

    conf_filepath = os.path.join(model_filepath, 'reduced-config.json')
    logging.info("Loading config file from: {}".format(conf_filepath))
    with open(conf_filepath, 'r') as fh:
        round_config = json.load(fh)

    logging.info("Loading model from filepath: {}".format(model_filepath))
    # https://huggingface.co/docs/transformers/installation#offline-mode
    if round_config['use_lora']:
        base_model_filepath = os.path.join(model_filepath, 'base-model')
        logging.info("loading the base model (before LORA) from {}".format(base_model_filepath))
        model = AutoModelForCausalLM.from_pretrained(base_model_filepath, device_map = "auto", trust_remote_code=True, torch_dtype=torch_dtype, local_files_only=True)
        # model = AutoModelForCausalLM.from_pretrained(round_config['model_architecture'], trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch_dtype)

        fine_tuned_model_filepath = os.path.join(model_filepath, 'fine-tuned-model')
        logging.info("loading the LORA adapter onto the base model from {}".format(fine_tuned_model_filepath))
        model.load_adapter(fine_tuned_model_filepath)
    else:
        fine_tuned_model_filepath = os.path.join(model_filepath, 'fine-tuned-model')
        logging.info("Loading full fine tune checkpoint into cpu from {}".format(fine_tuned_model_filepath))
        model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_filepath, device_map = "auto", trust_remote_code=True, torch_dtype=torch_dtype, local_files_only=True)
        # model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_filepath, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch_dtype)

    model.eval()

    tokenizer_filepath = os.path.join(model_filepath, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_filepath)

    return model, tokenizer

import os, json, logging, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def _two_gpu_max_memory(headroom_gb=2):
    """
    Reserve headroom so HF sharding MUST split across both 16GB T4s.
    """
    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    cap = f"{16 - headroom_gb}GiB"  # e.g., "14GiB"
    return {i: cap for i in range(n)}

def _common_from_pretrained_kwargs():
    """
    Settings that reduce both CPU and GPU peak memory and use a lean attention impl.
    """
    kw = dict(
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,     # T4 → FP16
        low_cpu_mem_usage=True,        # streaming load
        offload_state_dict=True,       # avoid CPU spikes
        attn_implementation="sdpa",    # available by default on Kaggle
    )
    mm = _two_gpu_max_memory(headroom_gb=2)
    if mm and torch.cuda.device_count() > 1:
        kw["device_map"] = "auto"
        kw["max_memory"] = mm
        # Optional if host RAM is tight:
        # kw["offload_folder"] = "/kaggle/working/offload"
    else:
        kw["device_map"] = {"": 0}
    return kw

def load_model_and_tokenizer(model_dir: str, merge_lora: bool = True):
    """
    Robust loader for full fine-tunes or LoRA adapters stored under `model_dir`.
    Expects:
      - reduced-config.json with {"use_lora": <bool>, ...}
      - For LoRA: base-model/, fine-tuned-model/
      - For full FT: fine-tuned-model/
      - tokenizer/ with tokenizer files
    Returns: (model, tokenizer)
    """
    conf_path = os.path.join(model_dir, "reduced-config.json")
    logging.info(f"Loading config: {conf_path}")
    with open(conf_path, "r") as fh:
        cfg = json.load(fh)

    kw = _common_from_pretrained_kwargs()

    if cfg.get("use_lora", False):
        base_dir = os.path.join(model_dir, "base-model")
        lora_dir = os.path.join(model_dir, "fine-tuned-model")

        logging.info(f"Loading base model: {base_dir}")
        model = AutoModelForCausalLM.from_pretrained(base_dir, **kw)
        logging.info(f"Attaching LoRA adapter: {lora_dir}")
        # If PeftModel is missing, use .load_adapter if available
        try:
            model = PeftModel.from_pretrained(model, lora_dir, is_trainable=False)  # type: ignore
        except Exception:
            model.load_adapter(lora_dir)

    else:
        ft_dir = os.path.join(model_dir, "fine-tuned-model")
        logging.info(f"Loading full fine-tuned model: {ft_dir}")
        model = AutoModelForCausalLM.from_pretrained(ft_dir, **kw)

    # Tokenizer hygiene
    tok_dir = os.path.join(model_dir, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # better for causal LMs with dynamic padding

    # Runtime memory knobs for your gradient-based rollout
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  # reduce KV/activation memory during your search

    # Optional: quick sanity check of sharding
    try:
        print(getattr(model, "hf_device_map", "no device map"))
    except Exception:
        pass

    return model, tokenizer

def download_and_load(file_id, output_filename, load_model_path):
    """
    Wrapper that uses your existing helpers:
      - clear_memory(), download_file_from_google_drive(), process_paths()
    """
    clear_memory(verbose=False)

    _ = download_file_from_google_drive(
        file_id=file_id,
        output_dir="/kaggle/tmp",
        output_filename=output_filename,
        quiet=False
    )

    process_paths(paths=["/kaggle/tmp"], recursive=True, max_depth=None)

    model, tokenizer = load_model_and_tokenizer(load_model_path, merge_lora=True)
    return model, tokenizer

