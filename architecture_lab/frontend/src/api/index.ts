import type { ModulesInfo, ModelConfig, ParamEstimate, TrainConfig, TrainMessage, Experiment, DefaultsResponse, GpuInfo, TrainSnapshotData } from '../types'

const BASE = '/api'

async function parseJsonOrNull(res: Response): Promise<any> {
  if (typeof res.text !== 'function') {
    return typeof res.json === 'function' ? res.json() : null
  }
  const text = await res.text()
  if (!text) {
    return null
  }
  return JSON.parse(text)
}

async function parseApiResponse<T>(res: Response): Promise<T> {
  const payload = await parseJsonOrNull(res)
  if (!res.ok) {
    throw new Error(payload?.error ?? `${res.status} ${res.statusText}`.trim())
  }
  return payload as T
}

export async function fetchModules(): Promise<ModulesInfo> {
  const res = await fetch(`${BASE}/modules`)
  return parseApiResponse<ModulesInfo>(res)
}

export async function fetchDefaults(): Promise<DefaultsResponse> {
  const res = await fetch(`${BASE}/defaults`)
  return parseApiResponse<DefaultsResponse>(res)
}

export async function fetchGpuInfo(): Promise<GpuInfo> {
  const res = await fetch(`${BASE}/gpu`)
  return parseApiResponse<GpuInfo>(res)
}

export async function fetchTrainStatus(): Promise<TrainSnapshotData> {
  const res = await fetch(`${BASE}/train/status`)
  return parseApiResponse<TrainSnapshotData>(res)
}

export async function fetchExperiments(): Promise<Experiment[]> {
  const res = await fetch(`${BASE}/experiments`)
  return parseApiResponse<Experiment[]>(res)
}

export async function estimateParams(config: ModelConfig): Promise<ParamEstimate> {
  const res = await fetch(`${BASE}/estimate_params`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  return parseApiResponse<ParamEstimate>(res)
}

export async function saveExperiment(experiment: Experiment): Promise<Experiment> {
  const res = await fetch(`${BASE}/experiments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: experiment }),
  })
  return parseApiResponse<Experiment>(res)
}

export async function updateExperimentName(experimentId: string, name: string): Promise<Experiment> {
  const res = await fetch(`${BASE}/experiments/${experimentId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: { name } }),
  })
  return parseApiResponse<Experiment>(res)
}

export async function deleteExperiment(experimentId: string): Promise<void> {
  const res = await fetch(`${BASE}/experiments/${experimentId}`, {
    method: 'DELETE',
  })
  await parseApiResponse(res)
}

export async function startTraining(
  config: ModelConfig,
  trainConfig: TrainConfig,
  options: { runName?: string; paramCount?: number } = {},
): Promise<{ status: string; run_id: string }> {
  const res = await fetch(`${BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_cfg: config,
      train_config: trainConfig,
      run_name: options.runName ?? null,
      param_count: options.paramCount ?? null,
    }),
  })
  return parseApiResponse<{ status: string; run_id: string }>(res)
}

export async function stopTraining(): Promise<void> {
  const res = await fetch(`${BASE}/stop`, { method: 'POST' })
  await parseApiResponse(res)
}

export async function clearFinishedTraining(runId: string): Promise<void> {
  const res = await fetch(`${BASE}/train/clear_finished`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId }),
  })
  await parseApiResponse(res)
}

export function connectTrainWebSocket(
  onMessage: (msg: TrainMessage) => void,
  options: { onClose?: (event: CloseEvent) => void; onError?: () => void } = {},
): Promise<WebSocket> {
  return new Promise((resolve, reject) => {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${location.host}/ws/train`)
    let settled = false

    ws.onopen = () => {
      settled = true
      resolve(ws)
    }
    ws.onmessage = (event) => {
      onMessage(JSON.parse(event.data))
    }
    ws.onerror = () => {
      options.onError?.()
      if (!settled) {
        reject(new Error('Failed to connect training WebSocket'))
      }
    }
    ws.onclose = (event) => {
      options.onClose?.(event)
      if (!settled) {
        reject(new Error('Training WebSocket closed before it was ready'))
      }
    }
  })
}
