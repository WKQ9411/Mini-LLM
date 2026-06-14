import test from 'node:test'
import assert from 'node:assert/strict'

import {
  applyLayerConfigToTargets,
  getParamValue,
  isParamLocked,
  parseTargetLayerIds,
  updateLayerParamValue,
} from './layerConfigState.js'

test('sliding deepseek v4 locks all compression and indexer params', () => {
  const layer = {
    attention_type: 'csa_hca',
    attention_params: {
      attention_mechanism: 'sliding',
      compress_ratio: 64,
      index_num_attention_heads: 4,
      index_head_dim: 32,
      index_topk: 8,
      index_score_bias_alpha: 1.5,
      compress_rope_theta: 40000,
    },
    ffn_type: 'swiglu',
    ffn_params: {},
  }

  const locked = ['compress_ratio', 'compress_rope_theta', 'index_num_attention_heads', 'index_head_dim', 'index_topk', 'index_score_bias_alpha']
  for (const key of locked) {
    assert.equal(isParamLocked(layer, 'attention_params', key), true, `${key} should be locked`)
    assert.equal(getParamValue(layer, 'attention_params', key, { default: 99 }), 0, `${key} should show 0`)
  }

  // sliding uses rope_theta, should NOT be locked
  assert.equal(isParamLocked(layer, 'attention_params', 'rope_theta'), false)
})

test('csa locks rope_theta and window_size', () => {
  const layer = {
    attention_type: 'csa_hca',
    attention_params: { attention_mechanism: 'csa' },
    ffn_type: 'swiglu',
    ffn_params: {},
  }

  assert.equal(isParamLocked(layer, 'attention_params', 'rope_theta'), true)
  assert.equal(isParamLocked(layer, 'attention_params', 'window_size'), true)
  assert.equal(getParamValue(layer, 'attention_params', 'window_size', { default: 32 }), 0)
  assert.equal(isParamLocked(layer, 'attention_params', 'index_topk'), false)
  assert.equal(isParamLocked(layer, 'attention_params', 'compress_ratio'), false)
})

test('csa_hca without explicit mechanism uses csa locking defaults', () => {
  const layer = {
    attention_type: 'csa_hca',
    attention_params: {},
    ffn_type: 'swiglu',
    ffn_params: {},
  }

  assert.equal(isParamLocked(layer, 'attention_params', 'window_size'), true)
  assert.equal(getParamValue(layer, 'attention_params', 'window_size', { default: 32 }), 0)
  assert.equal(isParamLocked(layer, 'attention_params', 'compress_ratio'), false)
})

test('hca locks indexer params, rope_theta, and window_size', () => {
  const layer = {
    attention_type: 'csa_hca',
    attention_params: { attention_mechanism: 'hca' },
    ffn_type: 'swiglu',
    ffn_params: {},
  }

  const locked = ['rope_theta', 'window_size', 'index_num_attention_heads', 'index_head_dim', 'index_topk', 'index_score_bias_alpha']
  for (const key of locked) {
    assert.equal(isParamLocked(layer, 'attention_params', key), true, `${key} should be locked for hca`)
  }

  assert.equal(isParamLocked(layer, 'attention_params', 'compress_ratio'), false)
  assert.equal(isParamLocked(layer, 'attention_params', 'compress_rope_theta'), false)
})

test('switching mechanism drops all newly-locked params', () => {
  const layer = {
    attention_type: 'csa_hca',
    attention_params: {
      attention_mechanism: 'csa',
      compress_ratio: 32,
      compress_rope_theta: 40000,
      index_num_attention_heads: 4,
      index_head_dim: 32,
      index_topk: 8,
      index_score_bias_alpha: 1.5,
      window_size: 64,
    },
    ffn_type: 'swiglu',
    ffn_params: {},
  }

  // csa -> sliding: all compression + indexer params dropped
  const toSliding = updateLayerParamValue(layer, 'attention_params', 'attention_mechanism', 'sliding')
  assert.equal('compress_ratio' in toSliding.attention_params, false)
  assert.equal('compress_rope_theta' in toSliding.attention_params, false)
  assert.equal('index_num_attention_heads' in toSliding.attention_params, false)
  assert.equal('index_head_dim' in toSliding.attention_params, false)
  assert.equal('index_topk' in toSliding.attention_params, false)
  assert.equal('index_score_bias_alpha' in toSliding.attention_params, false)

  // csa -> hca: indexer params + rope_theta dropped
  const toHca = updateLayerParamValue(layer, 'attention_params', 'attention_mechanism', 'hca')
  assert.equal('rope_theta' in toHca.attention_params, false)
  assert.equal('index_num_attention_heads' in toHca.attention_params, false)
  assert.equal('index_head_dim' in toHca.attention_params, false)
  assert.equal('index_topk' in toHca.attention_params, false)
  assert.equal('index_score_bias_alpha' in toHca.attention_params, false)
  assert.equal('window_size' in toHca.attention_params, false)
  assert.equal(toHca.attention_params.compress_ratio, 32) // kept
})

test('re-enabled mechanism params fall back to defaults instead of stale zeroes', () => {
  const layer = {
    attention_type: 'csa_hca',
    attention_params: {
      attention_mechanism: 'hca',
      rope_theta: 10000,
      index_num_attention_heads: 4,
      index_head_dim: 32,
    },
    ffn_type: 'swiglu',
    ffn_params: {},
  }

  const toHca = updateLayerParamValue(layer, 'attention_params', 'attention_mechanism', 'hca')
  const backToCsa = updateLayerParamValue(toHca, 'attention_params', 'attention_mechanism', 'csa')

  assert.equal(getParamValue(backToCsa, 'attention_params', 'index_num_attention_heads', { default: 2 }), 2)
  assert.equal(getParamValue(backToCsa, 'attention_params', 'index_head_dim', { default: 16 }), 16)
})

test('parseTargetLayerIds keeps only valid unique layer ids excluding source layer', () => {
  assert.deepEqual(
    parseTargetLayerIds('0, 2, 2, 7, foo, 1', 4, 1),
    [0, 2],
  )
})

test('applyLayerConfigToTargets copies a full layer configuration to specific targets', () => {
  const layers = [
    {
      attention_type: 'gated',
      attention_params: { num_heads: 8 },
      ffn_type: 'geglu',
      ffn_params: { intermediate_size: 512 },
    },
    {
      attention_type: 'standard',
      attention_params: { num_heads: 4 },
      ffn_type: 'swiglu',
      ffn_params: { intermediate_size: 256 },
    },
    {
      attention_type: 'mla',
      attention_params: { q_lora_rank: 32 },
      ffn_type: 'glu',
      ffn_params: { intermediate_size: 320 },
    },
  ]

  const nextLayers = applyLayerConfigToTargets(layers, 0, [2])

  assert.deepEqual(nextLayers[2], layers[0])
  assert.notEqual(nextLayers[2], layers[0])
  assert.deepEqual(nextLayers[1], layers[1])
})
