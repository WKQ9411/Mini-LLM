function getMechanism(layer) {
  if (layer.attention_type !== 'csa_hca') return null
  return layer.attention_params?.attention_mechanism || 'csa'
}

// 各 mechanism 下无用的参数
const LOCKED_BY_MECHANISM = {
  sliding: [
    'compress_ratio',
    'compress_rope_theta',
    'index_num_attention_heads',
    'index_head_dim',
    'index_topk',
    'index_score_bias_alpha',
  ],
  csa: [
    'rope_theta',
    'window_size',
  ],
  hca: [
    'rope_theta',
    'window_size',
    'index_num_attention_heads',
    'index_head_dim',
    'index_topk',
    'index_score_bias_alpha',
  ],
}

export function cloneLayerConfig(layer) {
  return JSON.parse(JSON.stringify(layer))
}

export function isParamLocked(layer, group, key) {
  if (group !== 'attention_params') return false
  const mechanism = getMechanism(layer)
  if (!mechanism) return false
  return LOCKED_BY_MECHANISM[mechanism]?.includes(key) ?? false
}

export function getParamValue(layer, group, key, param) {
  if (isParamLocked(layer, group, key)) {
    return 0
  }
  return layer[group]?.[key] ?? param.default
}

export function updateLayerParamValue(layer, group, key, value) {
  const nextGroup = {
    ...layer[group],
    [key]: value,
  }

  // Drop params disabled by the selected mechanism so re-enabled params use defaults.
  if (group === 'attention_params' && key === 'attention_mechanism') {
    const lockedKeys = LOCKED_BY_MECHANISM[value] ?? []
    for (const lk of lockedKeys) {
      delete nextGroup[lk]
    }
  }

  return {
    ...layer,
    [group]: nextGroup,
  }
}

export function parseTargetLayerIds(rawValue, layerCount, sourceIndex) {
  if (!rawValue.trim()) {
    return []
  }

  const tokens = rawValue.split(/[,\s]+/).filter(Boolean)
  const targets = []
  const seen = new Set()

  for (const token of tokens) {
    if (!/^\d+$/.test(token)) {
      continue
    }

    const layerId = Number(token)
    if (
      Number.isInteger(layerId) &&
      layerId >= 0 &&
      layerId < layerCount &&
      layerId !== sourceIndex &&
      !seen.has(layerId)
    ) {
      seen.add(layerId)
      targets.push(layerId)
    }
  }

  return targets
}

export function applyLayerConfigToTargets(layers, sourceIndex, targetIndices) {
  const sourceLayer = layers[sourceIndex]
  if (!sourceLayer) {
    return layers
  }

  const targetSet = new Set(targetIndices)
  return layers.map((layer, index) => (
    targetSet.has(index) ? cloneLayerConfig(sourceLayer) : layer
  ))
}
