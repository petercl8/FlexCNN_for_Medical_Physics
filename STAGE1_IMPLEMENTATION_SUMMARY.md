# Stage 1 Implementation Summary

## Overview
Implemented end-to-end attenuation data support with ordered parameters, nested tuple returns, and custom collation to handle None values.

## Key Design Decisions
1. **Ordered Parameters**: All helpers follow `sino, image, atten_sino, atten_image, recon1, recon2` ordering
2. **Nested Tuple Returns**: Dataloader returns `(act_data, atten_data, recon_data)` where:
   - `act_data = (sino, image)`
   - `atten_data = (atten_sino, atten_image)`
   - `recon_data = (recon1, recon2)`
3. **None Handling**: Any entry may be None (including `(None, None)` for activity data)
4. **No Attenuation Normalization**: Attenuation data is scaled only, not normalized
5. **Custom Collation**: `collate_nested()` function stacks tensors and preserves None values

## Files Modified

### 1. dataset_augment_data_recons.py
- **AugmentSinoImageDataRecons**: Reordered to accept `sino, image, atten_sino, atten_image, recon1, recon2`
- **AugmentImageImageDataRecons**: Same reordering
- All inner functions (`RandRotate*`, `*Flip*`, `ChannelFlip*`) updated with ordered params
- Attenuation data receives identical transforms as corresponding base types:
  - `atten_image` follows image transforms
  - `atten_sino` follows sinogram transforms (including roll and vertical flip)

### 2. dataset_resizing.py
- **resize_image_data()**: Now accepts `atten_image_multChannel` and returns 4-tuple
- **crop_pad_sino()**: Now accepts optional `atten_sino_multChannel` and returns 2-tuple
- Both functions apply identical transforms to attenuation data as their base types

### 3. dataset_classes.py
- **NpArrayDataLoader()**:
  - Added `atten_image_array` and `atten_sino_array` parameters
  - Extracts `atten_image_scale` and `atten_sino_scale` from settings
  - Converts attenuation arrays to tensors with None checks
  - Applies ordered augmentation with new signatures
  - Calls updated resizing functions with attenuation data
  - Applies scaling to attenuation (no normalization)
  - Returns nested tuple structure: `(act_data, atten_data, recon_data)`

- **NpArrayDataSet**:
  - Added `atten_image_path` and `atten_sino_path` parameters
  - Loads and slices attenuation arrays (or leaves as None)
  - Updated `__len__()` to handle None activity arrays
  - Updated `__getitem__()` to return nested tuples directly

### 4. run_supervisory.py
- **collate_nested()**: New function to handle nested tuples with None
  - Stacks tensors when all samples provide them
  - Returns None when all samples have None
  - Raises ValueError for mixed None/tensor (fail-fast)
- **DataLoader**: Now passes `collate_fn=collate_nested` and attenuation paths
- **Batch loop**: Updated to unpack `act_data, atten_data, recon_data`
  - `sino_scaled, act_map_scaled = act_data`
  - `atten_sino, atten_image = atten_data`
  - `recon1, recon2 = recon_data`

### 5. construct_dictionaries.py
- **setup_paths()**: Added attenuation file paths for all modes
  - `tune_atten_sino_path`, `tune_atten_image_path`
  - `train_atten_sino_path`, `train_atten_image_path`
  - `test_atten_sino_path`, `test_atten_image_path`
  - `visualize_atten_sino_path`, `visualize_atten_image_path`
  - Active paths set based on run_mode

- **setup_settings()**: Added attenuation scales
  - `atten_image_scale` (default 1.0)
  - `atten_sino_scale` (default 1.0)

## Usage

### To use attenuation data:
1. Add attenuation file names to `data_files` dict:
   ```python
   data_files = {
       'train_atten_sino_file': 'train_atten_sino.npy',
       'train_atten_image_file': 'train_atten_image.npy',
       # ... same for test, tune, visualize modes
   }
   ```

2. Optionally set attenuation scales in `common_settings`:
   ```python
   common_settings = {
       'atten_sino_scale': 1.0,
       'atten_image_scale': 1.0,
       # ... other settings
   }
   ```

3. Access attenuation in training loop:
   ```python
   for act_data, atten_data, recon_data in dataloader:
       sino, image = act_data
       atten_sino, atten_image = atten_data
       recon1, recon2 = recon_data
       # Use attenuation data as needed
   ```

### To skip attenuation (backward compatible):
- Simply don't provide attenuation file names in `data_files`
- Dataloader will return `atten_data = (None, None)`

## Testing Recommendations
1. Test with activity data only (no attenuation) to verify backward compatibility
2. Test with attenuation data to verify proper loading and transforms
3. Test with None activity data (`act_data = (None, None)`) for attenuation-only training
4. Verify augmentation synchronization across all data types
5. Test collate_fn with batches of varying None patterns

## Next Steps
To enable attenuation in your workflow:
1. Generate or locate attenuation sinogram and image files
2. Add file names to the notebook's `data_files` dict
3. Set appropriate scales if needed
4. Update loss functions/metrics to consume attenuation data when ready
