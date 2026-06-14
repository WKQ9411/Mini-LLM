import type { Experiment, LayerConfig } from '../types'

export function formatDuration(seconds: number | null | undefined): string

export function summarizeLayerStack(layers: LayerConfig[]): string

export function hydrateExperiment(experiment: Partial<Experiment> & Pick<Experiment, 'id' | 'name' | 'timestamp' | 'model_config' | 'train_config' | 'param_count' | 'loss_history'>, fallbackColorIndex?: number): Experiment

export function buildExperimentOverview(experiment: Experiment): {
  layerCount: number
  architectureSummary: string
  formattedElapsed: string
  formattedEta: string
}
