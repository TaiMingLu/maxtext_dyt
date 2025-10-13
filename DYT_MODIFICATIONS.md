# DyT (Dynamic Transformation) Modifications for MaxText

This document describes the DyT modifications implemented in this fork of MaxText, which add support for learnable dynamic normalization layers and embedding scaling.

## Overview of Modifications

The following three key modifications have been implemented:

### 1. **Configurable Normalization Types**

Added support for alternative normalization layers that can replace the default RMSNorm:

- **RMSNorm** (default): The standard Root Mean Square Normalization
- **DynamicTanh**: `scale * tanh(alpha * x)` where `alpha` is a learnable scalar parameter
- **DynamicErf** (ShiftedErf): `scale * erf(alpha * x + shift)` where both `alpha` and `shift` are learnable scalar parameters

Both DynamicTanh and DynamicErf have **no bias** term (as per the patch), and use a single scalar `alpha` (not per-dimension) following the DyT paper design.

### 2. **Learnable Embedding Scale**

Added a learnable `shared_scale` parameter that is applied to embeddings after the token and position embeddings are combined:

- Initialized to `sqrt(emb_dim)`
- Can be enabled/disabled via configuration
- Applied before dropout

### 3. **Separate Alpha Initialization for Different Layers**

Different normalization layers in the model can have different alpha initialization values:

- `attn_alpha_init_value`: For pre-attention normalization layers
- `ffn_alpha_init_value`: For pre-FFN (post-attention) normalization layers  
- `decoder_alpha_init_value`: For the final decoder normalization layer (decoder_norm)

## Configuration Parameters

Add these parameters to your config file (they're already in `base.yml`):

```yaml
# DyT Normalization Configuration
norm_type: 'rms'  # Options: 'rms', 'tanh', 'shifterf'

# Alpha initialization values for different normalization layers
attn_alpha_init_value: 1.0    # Pre-attention normalization
ffn_alpha_init_value: 1.0     # Pre-FFN normalization  
decoder_alpha_init_value: 1.0 # Final decoder normalization

# Shift initialization (only for shifterf)
shift_init_value: 0.0

# Shared scale for embeddings
use_shared_scale: False  # Set to True to enable
```

## Usage Examples

### Example 1: Train Llama 1B with DynamicTanh Normalization

```bash
python3 -m MaxText.train MaxText/configs/base.yml \
    model_name=llama3.1-1b \
    run_name=llama1b_dyt_tanh \
    base_output_directory=gs://YOUR_BUCKET \
    norm_type=tanh \
    attn_alpha_init_value=0.5 \
    ffn_alpha_init_value=0.5 \
    decoder_alpha_init_value=0.5 \
    use_shared_scale=True \
    steps=50000 \
    per_device_batch_size=8
```

### Example 2: Use Pre-configured DyT Model

We've created a pre-configured model config file:

```bash
python3 -m MaxText.train MaxText/configs/models/llama3.1-1b-dyt.yml \
    run_name=llama1b_dyt_experiment \
    base_output_directory=gs://YOUR_BUCKET \
    dataset_type=hf \
    hf_path='allenai/c4' \
    hf_data_dir='en' \
    steps=50000
```

### Example 3: Use DynamicErf (ShiftedErf) with Custom Alpha Values

```bash
python3 -m MaxText.train MaxText/configs/base.yml \
    model_name=llama3.1-1b \
    norm_type=shifterf \
    attn_alpha_init_value=0.1 \
    ffn_alpha_init_value=0.1 \
    decoder_alpha_init_value=0.1 \
    shift_init_value=0.0 \
    use_shared_scale=True
```

## Implementation Details

### File Modifications

1. **`MaxText/layers/normalizations.py`**
   - Added `DynamicTanh` class (nnx.Module)
   - Added `DynamicErf` class (nnx.Module)
   - Added convenience functions: `dynamic_tanh()`, `dynamic_erf()`
   - Added `create_norm_layer()` helper function that selects the appropriate norm type based on config

2. **`MaxText/configs/base.yml`**
   - Added all DyT-related configuration parameters with defaults

3. **`MaxText/layers/llama2.py`**
   - Modified pre-attention normalization to use `create_norm_layer()` with `attn_alpha_init_value`
   - Modified post-attention (pre-FFN) normalization to use `create_norm_layer()` with `ffn_alpha_init_value`

4. **`MaxText/layers/decoders.py`**
   - Modified `DecoderLayer` to use `create_norm_layer()` for pre-attention norm
   - Modified `Decoder.get_norm_layer()` to use `create_norm_layer()` for final decoder norm with `decoder_alpha_init_value`
   - Added `shared_scale` parameter in `_apply_embedding()` method

5. **`MaxText/configs/models/llama3.1-1b-dyt.yml`**
   - Created example configuration with DyT settings

### Key Design Decisions

1. **Scalar Alpha**: Following the DyT paper, alpha is a single scalar value (not per-dimension), initialized via `jnp.array([alpha_init_value])`

2. **No Bias**: DynamicTanh and DynamicErf do not include bias terms (unlike the nanoGPT patch which had optional bias)

3. **Backward Compatibility**: When `norm_type='rms'` (default), the behavior is identical to the original MaxText

4. **Separate Alpha Values**: Different layers can have different alpha initialization values, allowing fine-grained control over the normalization behavior

## Model Architecture Changes

For a Llama-style model with DyT enabled, the architecture becomes:

```
Input Tokens
  ↓
Token Embedding + Position Embedding
  ↓
× shared_scale (if use_shared_scale=True)  ← NEW
  ↓
Decoder Layers (16×):
  ├─ DynamicTanh/Erf(alpha=attn_alpha)  ← MODIFIED
  ├─ Self-Attention
  ├─ Add & DynamicTanh/Erf(alpha=ffn_alpha)  ← MODIFIED
  ├─ MLP
  └─ Add
  ↓
Final DynamicTanh/Erf(alpha=decoder_alpha)  ← MODIFIED
  ↓
Output Logits
```

## Verifying the Implementation

To verify that DyT modifications are active, check the model initialization output:

1. You should see parameters like:
   - `decoder/layers_N/pre_self_attention_norm/alpha`
   - `decoder/layers_N/post_self_attention_layer_norm/alpha`
   - `decoder/decoder_norm/alpha`
   - `decoder/shared_scale` (if enabled)

2. For `shifterf`, you should also see:
   - `decoder/layers_N/*/shift` parameters

## Performance Considerations

- **Memory**: DynamicTanh and DynamicErf have minimal memory overhead (one or two scalar parameters per norm layer)
- **Compute**: The computational cost is similar to RMSNorm
- **Training**: Alpha values are learnable and will be optimized during training

## Troubleshooting

### Issue: "AttributeError: 'Config' object has no attribute 'norm_type'"

**Solution**: Make sure you're using the updated `base.yml` config file or explicitly set `norm_type` in your command line arguments.

### Issue: Model parameters don't include alpha

**Solution**: Verify that `norm_type` is set to either `'tanh'` or `'shifterf'` (not `'rms'`).

### Issue: Shared scale not appearing

**Solution**: Set `use_shared_scale=True` in your configuration.

## References

- DyT Paper: [Add reference if available]
- nanoGPT patch: Provided by user as reference implementation

## Future Work

- [ ] Add support for other model architectures (Gemma, Mistral, etc.)
- [ ] Performance benchmarking comparing RMSNorm vs DynamicTanh vs DynamicErf
- [ ] Add WandB logging for alpha/shift parameter values during training
- [ ] Hyperparameter tuning for optimal alpha initialization values

