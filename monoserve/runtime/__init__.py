"""The MonoServe runtime around the fabric: model weights in the fabric's
layout, the paged KV cache, per-request state, and the lane programs that
execute a MoE transformer as tiles."""
