import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
import os
import json
import shutil

def sample_trim_resize(
    input_dir,
    input_file,
    output_dir,
    output_file,
    sample_division=1,
    remove_n=0,
    crop_h=None,
    crop_w=None,
    new_height=None,
    new_width=None,
    pool_h_factor=None,
    pool_w_factor=None,
    seed=42,
    log_file='log-SampleTrimResize.txt',
    skip_if_exists_vm=False,
    dry_run=False
):
    """
    Sample, optionally remove slices, center-crop, and apply pooling or resize, then copy a numpy dataset to Google Drive
    with verification and logging. Useful for preparing image or sinogram datasets with precise dimensions.

    Args:
        input_dir (str): Path to the directory containing the input numpy file.
        input_file (str): Name of the input numpy file (should end with .npy).
        output_dir (str): Path to the directory where output file will be saved (e.g., Google Drive path).
        output_file (str): Name of the output file.
        sample_division (int, default=1): Keep every N-th slice along the first axis (after optional removal).
        remove_n (int, default=0): Number of slices to randomly remove before sampling.
        crop_h (int or None, default=None): Height after center-cropping. Must be <= original height.
            If None, no vertical cropping is applied.
        crop_w (int or None, default=None): Width after center-cropping. Must be <= original width.
            If None, no horizontal cropping is applied.
        new_height (int or None, default=None): Height to resize cropped slices to. If None, uses crop_h or original height.
            Mutually exclusive with pool_h_factor.
        new_width (int or None, default=None): Width to resize cropped slices to. If None, uses crop_w or original width.
            Mutually exclusive with pool_w_factor.
        pool_h_factor (int or None, default=None): Vertical pooling factor. If specified, dimension is padded (via edge
            replication) to nearest multiple of factor, then reduced by average pooling. Mutually exclusive with new_height.
        pool_w_factor (int or None, default=None): Horizontal pooling factor. If specified, dimension is padded (via edge
            replication) to nearest multiple of factor, then reduced by average pooling. Mutually exclusive with new_width.
        seed (int, default=42): Random seed for reproducible slice removal and sampling.
        log_file (str, default='log-SampleTrimResize.txt'): Name of the log file written in the current working directory.
        skip_if_exists_vm (bool, default=False): If True, processing is skipped if the VM output file already exists.
        dry_run (bool, default=False): If True, perform parameter validation and print output shape without creating files.

    Returns:
        None. Writes the processed dataset to:
            - local VM path: '/content/output_file'
            - output_dir/output_file
        and verifies that the VM and Drive copies match.

    Notes:
        - Cropping is centered and deterministic.
        - Pooling (if applied) occurs after cropping with automatic replication padding to divisible dimensions.
          Mutually exclusive with resizing. Use pooling for integer downsampling, resizing for arbitrary dimensions.
        - Sampling and slice removal occur before cropping/pooling/resizing.
        - The function logs parameters and output shape for reproducibility.
        - Uses GPU if available for pooling/resizing operations.
        - Suitable for image datasets or sinograms where exact spatial dimensions are important.
    """

    # --- Clean filenames and paths ---
    input_file = input_file.lstrip('/')
    output_file = output_file.lstrip('/')
    input_path = os.path.join(input_dir, input_file)
    local_output_path = os.path.join("/content", output_file)
    drive_output_path = os.path.join(output_dir, output_file)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load input ---
    arr = np.load(input_path, mmap_mode='r')
    n_samples, channels, height, width = arr.shape

    # --- Validate crop sizes ---
    if crop_h is not None and crop_h > height:
        raise ValueError(f"crop_h ({crop_h}) exceeds input height ({height})")
    if crop_w is not None and crop_w > width:
        raise ValueError(f"crop_w ({crop_w}) exceeds input width ({width})")

    # --- Validate mutual exclusivity of pooling and resizing ---
    if (pool_h_factor is not None or pool_w_factor is not None) and (new_height is not None or new_width is not None):
        raise ValueError("Pooling (pool_h_factor/pool_w_factor) and resizing (new_height/new_width) are mutually exclusive")

    # --- Compute crop indices ---
    h_start, h_end = 0, height
    w_start, w_end = 0, width

    if crop_h is not None:
        h_start = (height - crop_h) // 2
        h_end   = h_start + crop_h

    if crop_w is not None:
        w_start = (width - crop_w) // 2
        w_end   = w_start + crop_w

    crop_h_eff = h_end - h_start
    crop_w_eff = w_end - w_start

    # --- Compute padding needed for pooling ---
    pad_h = 0
    pad_w = 0
    if pool_h_factor is not None:
        pad_h = (pool_h_factor - crop_h_eff % pool_h_factor) % pool_h_factor
    if pool_w_factor is not None:
        pad_w = (pool_w_factor - crop_w_eff % pool_w_factor) % pool_w_factor

    # --- RNG setup ---
    rng = np.random.default_rng(seed)

    # --- Random slice removal ---
    all_indices = np.arange(n_samples)
    keep_mask = np.ones(n_samples, dtype=bool)
    if remove_n > 0:
        remove_idx = rng.choice(n_samples, size=remove_n, replace=False)
        keep_mask[remove_idx] = False
    remaining_idx = all_indices[keep_mask]

    # --- Even sampling ---
    sampled_idx = remaining_idx[::sample_division]
    n_final = len(sampled_idx)

    # --- Output dimensions ---
    if pool_h_factor is not None:
        out_h = (crop_h_eff + pad_h) // pool_h_factor
    else:
        out_h = new_height if new_height is not None else crop_h_eff
    
    if pool_w_factor is not None:
        out_w = (crop_w_eff + pad_w) // pool_w_factor
    else:
        out_w = new_width if new_width is not None else crop_w_eff

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n📊 Final dataset shape: ({n_final}, {channels}, {out_h}, {out_w})")
    
    # --- Operation selection flags ---
    apply_pooling = pool_h_factor is not None or pool_w_factor is not None
    apply_resize = new_height is not None or new_width is not None
    
    # --- Dry run: show parameters and exit ---
    if dry_run:
        print(f"\n🔍 DRY RUN MODE - No files will be created")
        print(f"   Input:  {input_path}")
        print(f"   Output: {drive_output_path}")
        print(f"   Original shape: ({n_samples}, {channels}, {height}, {width})")
        print(f"   Samples removed: {remove_n}")
        print(f"   Sample division: {sample_division}")
        print(f"   Crop (H×W): {crop_h_eff}×{crop_w_eff}")
        if apply_pooling:
            print(f"   Pooling: {pool_h_factor or 1}×{pool_w_factor or 1}")
            print(f"   Padding (H×W): {pad_h}×{pad_w}")
        elif apply_resize:
            print(f"   Resize to: {out_h}×{out_w}")
        print(f"   Final shape: ({n_final}, {channels}, {out_h}, {out_w})")
        return

    # --- Define resize_slice function ---
    if apply_pooling:
        def resize_slice(slice_tensor):
            # slice_tensor input: [C, H, W]
            slice_tensor = slice_tensor.unsqueeze(0)  # [1, C, H, W]
            if pad_h > 0 or pad_w > 0:
                slice_tensor = torch.nn.functional.pad(slice_tensor, (0, pad_w, 0, pad_h), mode='replicate')
            kernel_h = pool_h_factor or 1
            kernel_w = pool_w_factor or 1
            slice_tensor = torch.nn.functional.avg_pool2d(slice_tensor, kernel_size=(kernel_h, kernel_w))
            slice_tensor = slice_tensor.squeeze(0)  # [C, H, W]
            return slice_tensor
    elif apply_resize:
        resize_slice = T.Resize((out_h, out_w), antialias=True)
    else:
        resize_slice = lambda x: x  # identity function

    # --- Process slices ---
    if skip_if_exists_vm and os.path.exists(local_output_path):
        print(f"VM file already exists: {local_output_path}. Skipping processing slices...")
    else:
        result = np.lib.format.open_memmap(
            local_output_path,
            mode='w+',
            dtype=np.float32,
            shape=(n_final, channels, out_h, out_w)
        )

        tqdm_bar = tqdm(sampled_idx, desc="Processing slices")
        for i, idx in enumerate(tqdm_bar):
            slice_i = arr[idx]  # [C,H,W]
            slice_i = slice_i[:, h_start:h_end, w_start:w_end]

            slice_tensor = torch.from_numpy(slice_i).to(device)
            slice_tensor = resize_slice(slice_tensor)
            slice_i = slice_tensor.cpu().numpy().astype(np.float32)

            result[i] = slice_i

        print("Flushing memory-mapped array to disk...")
        result.flush()

    # --- Copy to Drive ---
    if os.path.abspath(local_output_path) != os.path.abspath(drive_output_path):
        print(f"Copying output to Drive: {drive_output_path}")
        shutil.copy(local_output_path, drive_output_path)

    # --- Verify ---
    vm_arr = np.load(local_output_path, mmap_mode='r')
    drive_arr = np.load(drive_output_path, mmap_mode='r')

    if vm_arr.shape == drive_arr.shape and vm_arr.dtype == drive_arr.dtype:
        print(f"\n✅ File verification successful!")
        print(f"   Shape: {vm_arr.shape}, dtype: {vm_arr.dtype}")
    else:
        print("\n⚠️ File verification failed!")

    # --- Log ---
    run_entry = {
        "input_dir": input_dir,
        "input_file": input_file,
        "output_dir": output_dir,
        "output_file": output_file,
        "sample_division": sample_division,
        "remove_n": remove_n,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "new_height": new_height,
        "new_width": new_width,
        "pool_h_factor": pool_h_factor,
        "pool_w_factor": pool_w_factor,
        "output_shape": vm_arr.shape,
        "seed": seed
    }

    log_path = os.path.join(os.getcwd(), log_file)
    with open(log_path, 'a') as f:
        f.write(json.dumps(run_entry) + "\n")

    print(f"Log updated: {os.path.abspath(log_file)}")
