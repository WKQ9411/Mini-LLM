export interface NumericParamInfo {
  type: 'int' | 'float'
  default: number
  min: number
  max: number
}

export interface SelectParamInfo {
  type: 'select'
  default: string
  options: string[]
}

export type ParamInfo = NumericParamInfo | SelectParamInfo

export interface ModuleTypeInfo {
  name: string
  params: Record<string, ParamInfo>
}

export interface ModulesInfo {
  attention_types: Record<string, ModuleTypeInfo>
  ffn_types: Record<string, ModuleTypeInfo>
}

export interface DefaultsResponse {
  model: {
    vocab_size: number
  }
}

export interface LayerConfig {
  attention_type: string
  attention_params: Record<string, number | string>
  ffn_type: string
  ffn_params: Record<string, number | string>
}

export interface ModelConfig {
  hidden_size: number
  vocab_size: number
  max_seq_len: number
  rms_norm_eps: number
  share_embedding_head: boolean
  layers: LayerConfig[]
}

export interface ParamEstimate {
  param_count: number
  embedding_param_count: number
  lm_head_param_count: number
  remaining_param_count: number
  share_embedding_head: boolean
}

export interface GpuDeviceInfo {
  index: number
  name: string
  memory_total_mb: number | null
  memory_used_mb: number | null
  memory_usage_pct: number | null
  utilization_pct: number | null
  temperature_c: number | null
}

export interface GpuInfo {
  available: boolean
  source: 'nvidia-smi' | 'torch' | 'none'
  message: string | null
  gpus: GpuDeviceInfo[]
  updated_at: number
}

export interface TrainConfig {
  batch_size: number
  learning_rate: number
  max_steps: number | null
  warmup_steps: number
  data_order_seed: number
  model_init_seed: number
}

export interface StepData {
  type: 'step'
  run_id?: string | null
  step: number
  target_total_steps: number
  dataset_total_steps: number
  progress_pct: number
  elapsed_seconds: number
  eta_seconds: number | null
  loss: number
  lr: number
  tokens_per_sec: number
}

export interface DoneData {
  type: 'done'
  run_id?: string | null
  final_loss: number | null
  target_total_steps: number
  dataset_total_steps: number
  elapsed_seconds: number
  total_steps: number
  stopped_early: boolean
}

export interface ErrorData {
  type: 'error'
  run_id?: string | null
  message: string
}

export interface TrainSnapshotData {
  type: 'snapshot'
  run_id: string | null
  status: 'idle' | 'running' | 'stopping' | 'done' | 'error'
  training: boolean
  loss_history: Array<{ step: number; loss: number; lr: number }>
  progress: StepData | null
  done: DoneData | null
  error: ErrorData | null
  model_config: ModelConfig | null
  train_config: TrainConfig | null
  run_name: string | null
  param_count: number | null
  started_at: number | null
  updated_at: number | null
}

export type TrainMessage = StepData | DoneData | ErrorData | TrainSnapshotData

export interface Experiment {
  id: string
  name: string
  colorIndex?: number
  timestamp: number
  model_config: ModelConfig
  train_config: TrainConfig
  param_count: number
  loss_history: Array<{ step: number; loss: number; lr: number }>
  final_loss?: number | null
  completed_steps: number
  target_total_steps: number
  dataset_total_steps: number | null
  elapsed_seconds: number | null
}

export interface TrainingProgress {
  step: number
  targetTotalSteps: number
  datasetTotalSteps: number | null
  progressPct: number
  elapsedSeconds: number
  etaSeconds: number | null
  loss: number
  lr: number
  tokensPerSec: number
}
