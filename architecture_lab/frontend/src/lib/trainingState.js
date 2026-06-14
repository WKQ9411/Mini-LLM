export function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) {
    return '--'
  }

  const wholeSeconds = Math.max(0, Math.round(seconds))
  const hours = Math.floor(wholeSeconds / 3600)
  const minutes = Math.floor((wholeSeconds % 3600) / 60)
  const secs = wholeSeconds % 60

  if (hours > 0) {
    return [hours, minutes, secs].map((value) => String(value).padStart(2, '0')).join(':')
  }
  return [minutes, secs].map((value) => String(value).padStart(2, '0')).join(':')
}

export function summarizeLayerStack(layers) {
  if (!Array.isArray(layers) || layers.length === 0) {
    return 'No layers'
  }

  const groups = []
  for (const layer of layers) {
    const signature = `${layer.attention_type} / ${layer.ffn_type}`
    const last = groups[groups.length - 1]
    if (last?.signature === signature) {
      last.count += 1
    } else {
      groups.push({ signature, count: 1 })
    }
  }

  return groups.map((group) => `${group.count}x ${group.signature}`).join(', ')
}

export function hydrateExperiment(experiment, fallbackColorIndex = 0) {
  const completedSteps = experiment.completed_steps ?? experiment.loss_history?.length ?? 0
  return {
    ...experiment,
    colorIndex: experiment.colorIndex ?? fallbackColorIndex,
    completed_steps: completedSteps,
    target_total_steps: experiment.target_total_steps ?? experiment.train_config?.max_steps ?? completedSteps,
    dataset_total_steps: experiment.dataset_total_steps ?? null,
    elapsed_seconds: experiment.elapsed_seconds ?? null,
  }
}

export function buildExperimentOverview(experiment) {
  return {
    layerCount: experiment.model_config.layers.length,
    architectureSummary: summarizeLayerStack(experiment.model_config.layers),
    formattedElapsed: formatDuration(experiment.elapsed_seconds),
    formattedEta: '--',
  }
}
