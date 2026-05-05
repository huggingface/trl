## Summary

Reduce the config diff between `tiny-Cohere2ForCausalLM` and the reference `CohereLabs/tiny-aya-earth` by mirroring non-size config fields:

- `vocab_size=262144` (was `len(tokenizer.vocab)=261008`)
- `logit_scale=1.0`, `rope_theta=50000`, `bos_token_id=2`, `eos_token_id=3`
- 10 legacy kwargs stored in the ref config (`cache_implementation`, `layer_switch`, `order_of_interleaved_layers`, `position_embedding_type`, `rotary_pct`, `use_embedding_sharing`, `use_gated_activation`, `use_parallel_block`, `use_parallel_embedding`, `use_qk_norm`)

Remaining diffs are intentional size reductions (`head_dim`, `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_hidden_layers`, `num_key_value_heads`) plus `layer_types` (length tied to `num_hidden_layers`).

## Before

```
[config_diff] CohereLabs/tiny-aya-earth vs tiny (22 differences)
  bos_token_id                                     2                                  → 5
  cache_implementation                             hybrid                             → <missing>
  eos_token_id                                     3                                  → 255001
  head_dim                                         128                                → 2
  hidden_size                                      2048                               → 8
  intermediate_size                                11008                              → 32
  layer_switch                                     4                                  → <missing>
  layer_types                                      ['sliding_attention', 'sliding_att → ['sliding_attention', 'sliding_att
  logit_scale                                      1.0                                → 0.0625
  num_attention_heads                              16                                 → 4
  num_hidden_layers                                36                                 → 2
  num_key_value_heads                              4                                  → 2
  order_of_interleaved_layers                      local_attn_first                   → <missing>
  position_embedding_type                          rope_gptj                          → <missing>
  rope_theta                                       50000                              → 10000.0
  rotary_pct                                       1.0                                → <missing>
  use_embedding_sharing                            True                               → <missing>
  use_gated_activation                             True                               → <missing>
  use_parallel_block                               True                               → <missing>
  use_parallel_embedding                           False                              → <missing>
  use_qk_norm                                      False                              → <missing>
  vocab_size                                       262144                             → 261008
```

## After

```
[config_diff] CohereLabs/tiny-aya-earth vs tiny (7 differences)
  head_dim                                         128                                → 2
  hidden_size                                      2048                               → 8
  intermediate_size                                11008                              → 32
  layer_types                                      ['sliding_attention', 'sliding_att → ['sliding_attention', 'sliding_att
  num_attention_heads                              16                                 → 4
  num_hidden_layers                                36                                 → 2
  num_key_value_heads                              4                                  → 2
```
