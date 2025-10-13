#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Normalization Layers."""

from typing import Any, Tuple, Optional

from flax import linen as nn
from flax import nnx
from jax import lax
import jax
import jax.numpy as jnp
from MaxText import max_logging
from MaxText.layers import nnx_wrappers
from MaxText.layers.initializers import Initializer, variable_to_logically_partitioned


class RMSNorm(nnx.Module):
  """RMS normalization."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-6,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_axes: Tuple[Optional[str], ...] = (),
      scale_init: Initializer = nn.initializers.ones,
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.scale = nnx.Param(
        scale_init(rngs.params(), (num_features,), weight_dtype),
        sharding=kernel_axes,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = self.scale.value
    # Move scale to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving scale parameter to device")
      scale = jax.device_put(scale, jax._src.sharding_impls.TransferToMemoryKind("device"))

    scale = jnp.asarray(scale, self.dtype)
    return y * scale


def rms_norm(
    num_features: int,
    epsilon: float = 1e-6,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    kernel_axes: Tuple[Optional[str], ...] = (),
    scale_init: Initializer = nn.initializers.ones,
    name: Optional[str] = None,
    parameter_memory_host_offload: bool = False,
):
  """Creates a RMSNorm module."""
  module = nnx_wrappers.to_linen(
      RMSNorm,
      num_features=num_features,
      epsilon=epsilon,
      dtype=dtype,
      weight_dtype=weight_dtype,
      kernel_axes=kernel_axes,
      scale_init=scale_init,
      parameter_memory_host_offload=parameter_memory_host_offload,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
  )
  return module


class DynamicTanh(nnx.Module):
  """Dynamic Tanh normalization as replacement for RMSNorm."""

  def __init__(
      self,
      num_features: int,
      alpha_init_value: float = 1.0,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_axes: Tuple[Optional[str], ...] = (),
      scale_init: Initializer = nn.initializers.ones,
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.alpha_init_value = alpha_init_value
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.parameter_memory_host_offload = parameter_memory_host_offload
    
    # DyT paper uses a single scalar alpha, not per-dimension
    self.alpha = nnx.Param(
        jnp.array([alpha_init_value], dtype=weight_dtype),
        sharding=(),
    )
    self.scale = nnx.Param(
        scale_init(rngs.params(), (num_features,), weight_dtype),
        sharding=kernel_axes,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies dynamic tanh normalization on the input."""
    alpha = self.alpha.value
    scale = self.scale.value
    
    # Move parameters to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving DynamicTanh parameters to device")
      alpha = jax.device_put(alpha, jax._src.sharding_impls.TransferToMemoryKind("device"))
      scale = jax.device_put(scale, jax._src.sharding_impls.TransferToMemoryKind("device"))
    
    alpha = jnp.asarray(alpha, self.dtype)
    scale = jnp.asarray(scale, self.dtype)
    
    # Apply: scale * tanh(alpha * x)
    y = jnp.tanh(alpha * x)
    return jnp.asarray(y * scale, self.dtype)


def dynamic_tanh(
    num_features: int,
    alpha_init_value: float = 1.0,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    kernel_axes: Tuple[Optional[str], ...] = (),
    scale_init: Initializer = nn.initializers.ones,
    name: Optional[str] = None,
    parameter_memory_host_offload: bool = False,
):
  """Creates a DynamicTanh module."""
  module = nnx_wrappers.to_linen(
      DynamicTanh,
      num_features=num_features,
      alpha_init_value=alpha_init_value,
      dtype=dtype,
      weight_dtype=weight_dtype,
      kernel_axes=kernel_axes,
      scale_init=scale_init,
      parameter_memory_host_offload=parameter_memory_host_offload,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
  )
  return module


class DynamicErf(nnx.Module):
  """Dynamic Erf (Shifted Erf) normalization as replacement for RMSNorm."""

  def __init__(
      self,
      num_features: int,
      alpha_init_value: float = 1.0,
      shift_init_value: float = 0.0,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      kernel_axes: Tuple[Optional[str], ...] = (),
      scale_init: Initializer = nn.initializers.ones,
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.alpha_init_value = alpha_init_value
    self.shift_init_value = shift_init_value
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.parameter_memory_host_offload = parameter_memory_host_offload
    
    # DyT paper uses single scalar alpha and shift, not per-dimension
    self.alpha = nnx.Param(
        jnp.array([alpha_init_value], dtype=weight_dtype),
        sharding=(),
    )
    self.shift = nnx.Param(
        jnp.array([shift_init_value], dtype=weight_dtype),
        sharding=(),
    )
    self.scale = nnx.Param(
        scale_init(rngs.params(), (num_features,), weight_dtype),
        sharding=kernel_axes,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies dynamic erf normalization on the input."""
    alpha = self.alpha.value
    shift = self.shift.value
    scale = self.scale.value
    
    # Move parameters to device if parameter offloading is enabled
    if self.parameter_memory_host_offload:
      max_logging.log("normalizations.py: Moving DynamicErf parameters to device")
      alpha = jax.device_put(alpha, jax._src.sharding_impls.TransferToMemoryKind("device"))
      shift = jax.device_put(shift, jax._src.sharding_impls.TransferToMemoryKind("device"))
      scale = jax.device_put(scale, jax._src.sharding_impls.TransferToMemoryKind("device"))
    
    alpha = jnp.asarray(alpha, self.dtype)
    shift = jnp.asarray(shift, self.dtype)
    scale = jnp.asarray(scale, self.dtype)
    
    # Apply: scale * erf(alpha * x + shift)
    y = lax.erf(alpha * x + shift)
    return jnp.asarray(y * scale, self.dtype)


def dynamic_erf(
    num_features: int,
    alpha_init_value: float = 1.0,
    shift_init_value: float = 0.0,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    kernel_axes: Tuple[Optional[str], ...] = (),
    scale_init: Initializer = nn.initializers.ones,
    name: Optional[str] = None,
    parameter_memory_host_offload: bool = False,
):
  """Creates a DynamicErf module."""
  module = nnx_wrappers.to_linen(
      DynamicErf,
      num_features=num_features,
      alpha_init_value=alpha_init_value,
      shift_init_value=shift_init_value,
      dtype=dtype,
      weight_dtype=weight_dtype,
      kernel_axes=kernel_axes,
      scale_init=scale_init,
      parameter_memory_host_offload=parameter_memory_host_offload,
      name=name,
      metadata_fn=variable_to_logically_partitioned,
  )
  return module


def create_norm_layer(
    config,
    num_features: int,
    alpha_init_value: float = 1.0,
    dtype: Any = jnp.float32,
    weight_dtype: Any = jnp.float32,
    kernel_axes: Tuple[Optional[str], ...] = (),
    scale_init: Initializer = nn.initializers.ones,
    name: Optional[str] = None,
    parameter_memory_host_offload: bool = False,
):
  """Creates a normalization layer based on config.norm_type.
  
  Args:
    config: Model configuration containing norm_type and initialization values
    num_features: Number of features to normalize
    alpha_init_value: Alpha initialization value (for DyT norms)
    dtype: Data type for computations
    weight_dtype: Data type for parameters
    kernel_axes: Sharding axes for the scale parameter
    scale_init: Initializer for the scale parameter
    name: Optional name for the layer
    parameter_memory_host_offload: Whether to offload parameters to host memory
    
  Returns:
    A normalization layer module (RMSNorm, DynamicTanh, or DynamicErf)
  """
  norm_type = getattr(config, 'norm_type', 'rms')
  
  if norm_type == 'tanh':
    return dynamic_tanh(
        num_features=num_features,
        alpha_init_value=alpha_init_value,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_axes=kernel_axes,
        scale_init=scale_init,
        name=name,
        parameter_memory_host_offload=parameter_memory_host_offload,
    )
  elif norm_type == 'shifterf':
    shift_value = getattr(config, 'shift_init_value', 0.0)
    return dynamic_erf(
        num_features=num_features,
        alpha_init_value=alpha_init_value,
        shift_init_value=shift_value,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_axes=kernel_axes,
        scale_init=scale_init,
        name=name,
        parameter_memory_host_offload=parameter_memory_host_offload,
    )
  else:  # default to 'rms'
    epsilon = getattr(config, 'normalization_layer_epsilon', 1e-6)
    return rms_norm(
        num_features=num_features,
        epsilon=epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_axes=kernel_axes,
        scale_init=scale_init,
        name=name,
        parameter_memory_host_offload=parameter_memory_host_offload,
    )
