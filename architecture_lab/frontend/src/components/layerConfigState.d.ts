import type { LayerConfig, ParamInfo } from '../types'

export function cloneLayerConfig(layer: LayerConfig): LayerConfig

export function isParamLocked(
  layer: LayerConfig,
  group: 'attention_params' | 'ffn_params',
  key: string,
): boolean

export function getParamValue(
  layer: LayerConfig,
  group: 'attention_params' | 'ffn_params',
  key: string,
  param: ParamInfo,
): number | string

export function updateLayerParamValue(
  layer: LayerConfig,
  group: 'attention_params' | 'ffn_params',
  key: string,
  value: number | string,
): LayerConfig

export function parseTargetLayerIds(
  rawValue: string,
  layerCount: number,
  sourceIndex: number,
): number[]

export function applyLayerConfigToTargets(
  layers: LayerConfig[],
  sourceIndex: number,
  targetIndices: number[],
): LayerConfig[]
