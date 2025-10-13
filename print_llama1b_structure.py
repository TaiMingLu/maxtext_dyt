#!/usr/bin/env python3
"""
Script: Print Llama 3.1 1B Model Structure
"""

import yaml
import os


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def print_model_structure():
    """Print Llama 1B model structure"""
    
    # Load configuration
    config_path = "MaxText/configs/models/llama3.1-1b.yml"
    config_dict = load_config(config_path)
    
    print("=" * 80)
    print("Llama 3.1 1B Model Configuration")
    print("=" * 80)
    
    # Print configuration information
    for key, value in config_dict.items():
        print(f"{key:.<40} {value}")
    
    print("\n" + "=" * 80)
    print("Model Structure Details")
    print("=" * 80)
    
    # Calculate model parameters
    emb_dim = config_dict['base_emb_dim']
    num_heads = config_dict['base_num_query_heads']
    num_kv_heads = config_dict['base_num_kv_heads']
    num_layers = config_dict['base_num_decoder_layers']
    mlp_dim = config_dict['base_mlp_dim']
    head_dim = config_dict['head_dim']
    vocab_size = config_dict['vocab_size']
    
    print(f"\n[Overall Architecture]")
    print(f"  Model Type: Transformer Decoder (Autoregressive)")
    print(f"  Decoder Block Type: {config_dict['decoder_block']}")
    print(f"  Total Layers: {num_layers}")
    
    print(f"\n[Embedding Layer]")
    print(f"  Vocabulary Size: {vocab_size:,}")
    print(f"  Embedding Dimension: {emb_dim}")
    print(f"  Parameters: {vocab_size * emb_dim:,}")
    
    print(f"\n[Decoder Layer Structure] (Total {num_layers} layers)")
    print(f"  Each layer contains:")
    print(f"    1. RMS Normalization (Pre-Attention)")
    print(f"       - Normalization Dimension: {emb_dim}")
    print(f"       - epsilon: {config_dict['normalization_layer_epsilon']}")
    
    print(f"\n    2. Multi-Head Attention (Grouped Query Attention)")
    print(f"       - Number of Query Heads: {num_heads}")
    print(f"       - Number of Key/Value Heads: {num_kv_heads}")
    print(f"       - Dimension per Head: {head_dim}")
    print(f"       - Total Dimension: {num_heads * head_dim}")
    print(f"       - Q Projection Parameters: {emb_dim * num_heads * head_dim:,}")
    print(f"       - K Projection Parameters: {emb_dim * num_kv_heads * head_dim:,}")
    print(f"       - V Projection Parameters: {emb_dim * num_kv_heads * head_dim:,}")
    print(f"       - O Projection Parameters: {num_heads * head_dim * emb_dim:,}")
    print(f"       - Attention Total Parameters: {emb_dim * (num_heads + 2 * num_kv_heads) * head_dim + num_heads * head_dim * emb_dim:,}")
    
    print(f"\n    3. RMS Normalization (Post-Attention)")
    print(f"       - Normalization Dimension: {emb_dim}")
    
    print(f"\n    4. Feed-Forward Network (MLP)")
    print(f"       - Input Dimension: {emb_dim}")
    print(f"       - Hidden Dimension: {mlp_dim}")
    print(f"       - Output Dimension: {emb_dim}")
    print(f"       - Activation Function: {config_dict['mlp_activations']}")
    print(f"       - Gate Projection Parameters: {emb_dim * mlp_dim:,}")
    print(f"       - Up Projection Parameters: {emb_dim * mlp_dim:,}")
    print(f"       - Down Projection Parameters: {mlp_dim * emb_dim:,}")
    print(f"       - MLP Total Parameters: {2 * emb_dim * mlp_dim + mlp_dim * emb_dim:,}")
    
    # Calculate per-layer parameters
    attention_params = emb_dim * (num_heads + 2 * num_kv_heads) * head_dim + num_heads * head_dim * emb_dim
    mlp_params = 2 * emb_dim * mlp_dim + mlp_dim * emb_dim
    norm_params = emb_dim * 2  # Two RMS Norm layers
    layer_params = attention_params + mlp_params + norm_params
    
    print(f"\n    Total Parameters per Layer: {layer_params:,}")
    
    print(f"\n[Final RMSNorm]")
    print(f"  After all decoder layers, before output layer")
    print(f"  Name: decoder_norm")
    print(f"  Normalization Dimension: {emb_dim}")
    print(f"  Parameters: {emb_dim}")
    
    print(f"\n[Output Layer]")
    if config_dict.get('logits_via_embedding', False):
        print(f"  Using embedding weight sharing")
        output_params = 0
    else:
        print(f"  Independent linear projection layer")
        print(f"  Parameters: {emb_dim * vocab_size:,}")
        output_params = emb_dim * vocab_size
    
    print(f"\n[Position Encoding]")
    print(f"  Type: RoPE (Rotary Position Embedding)")
    print(f"  Maximum Time Scale: {config_dict['rope_max_timescale']:,}")
    
    print(f"\n[Total Parameter Statistics]")
    embedding_params = vocab_size * emb_dim
    decoder_params = layer_params * num_layers
    final_norm_params = emb_dim
    total_params = embedding_params + decoder_params + final_norm_params + output_params
    
    print(f"  Embedding Layer Parameters: {embedding_params:,}")
    print(f"  Decoder Parameters ({num_layers} layers): {decoder_params:,}")
    print(f"  Final RMSNorm Parameters: {final_norm_params:,}")
    print(f"  Output Layer Parameters: {output_params:,}")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Approximately {total_params / 1e9:.2f}B parameters")
    
    print("\n" + "=" * 80)
    print("Model Hierarchy")
    print("=" * 80)
    
    total_norms = num_layers * 2 + 1
    print(f"""
Transformer
├── Token Embedding ({vocab_size:,} × {emb_dim})
│
├── Decoder (Stack of {num_layers} layers)
│   ├── Layer 1
│   │   ├── RMS Norm (pre_self_attention_layer_norm)
│   │   ├── Self-Attention (Q:{num_heads}h, KV:{num_kv_heads}h, dim:{head_dim})
│   │   ├── Residual Connection
│   │   ├── RMS Norm (post_self_attention_layer_norm / pre-FFN)
│   │   ├── MLP (SwiGLU: {emb_dim} → {mlp_dim} → {emb_dim})
│   │   └── Residual Connection
│   ├── Layer 2
│   │   └── ... (same structure, 2 RMS Norms per layer)
│   ⋮
│   └── Layer {num_layers}
│       └── ... (same structure, 2 RMS Norms per layer)
│
├── Final RMS Norm (decoder_norm) ← The final normalization layer!
│
└── Output Layer ({emb_dim} × {vocab_size:,})

Total RMSNorm layers: {num_layers} layers × 2 + 1 final = {total_norms} RMSNorm layers
""")
    
    print("=" * 80)
    print("Key Features")
    print("=" * 80)
    
    print(f"""
✓ Grouped Query Attention (GQA): {num_kv_heads} KV heads shared across {num_heads} Query heads
✓ SwiGLU Activation Function: {config_dict['mlp_activations']}
✓ RMS Normalization: epsilon = {config_dict['normalization_layer_epsilon']}
  - 2 RMSNorm per layer (pre-attention + pre-FFN)
  - 1 Final RMSNorm (decoder_norm)
  - Total: {total_norms} RMSNorm layers
✓ RoPE Position Encoding: Maximum time scale {config_dict['rope_max_timescale']:,}
✓ Residual Connections: After each sublayer
✓ Dropout: {'Enabled' if config_dict['enable_dropout'] else 'Disabled'}
""")
    
    print("=" * 80)


if __name__ == "__main__":
    print_model_structure()

