<template>
  <div class="app">
    <div class="left-panel">
      <h2 class="brand-title">
        <img src="/logo.png" alt="Architecture Lab logo" class="brand-logo" />
        <span>Architecture Lab</span>
      </h2>
      <GlobalParams :model="modelConfig" @update:model="m => modelConfig = m" />
      <LayerConfig :layers="modelConfig.layers" :modules="modules" @update:layers="updateLayers" />
    </div>

    <div class="main-column">
      <LossChart :current-loss="displayedLoss" :experiments="selectedExperiments" :current-training-color-index="getNextExperimentColorIndex()" />
      <ComparePanel
        :experiments="savedExperiments"
        :selected-ids="selectedExpIds"
        :detail-id="detailExperimentId"
        @select="onSelectExperiments"
        @detail="openExperimentDetails"
        @delete="deleteExperiment"
        @delete-selected="deleteSelectedExperiments"
        @rename="renameExperiment"
      />
    </div>

    <div class="control-column">
      <ModelSummary :estimate="paramEstimate" />
      <GpuMonitor
        :info="gpuInfo"
        :loading="gpuLoading"
        :error="gpuError"
      />
      <TrainPanel
        v-model:config="trainConfig"
        v-model:run-name="pendingExperimentName"
        :training="training"
        :stopping="stopping"
        :progress="trainingProgress"
        @start="startTrain"
        @stop="stopTrain"
      />
    </div>
  </div>

  <ExperimentDetailPanel
    :open="detailExperimentOpen"
    :experiment="detailExperiment"
    @close="closeExperimentDetails"
  />
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted, computed } from 'vue'
import GlobalParams from './components/GlobalParams.vue'
import LayerConfig from './components/LayerConfig.vue'
import ModelSummary from './components/ModelSummary.vue'
import GpuMonitor from './components/GpuMonitor.vue'
import TrainPanel from './components/TrainPanel.vue'
import LossChart from './components/LossChart.vue'
import ComparePanel from './components/ComparePanel.vue'
import ExperimentDetailPanel from './components/ExperimentDetailPanel.vue'
import * as api from './api'
import type { ModelConfig, TrainConfig, TrainMessage, Experiment, ModulesInfo, ParamEstimate, TrainingProgress, GpuInfo, StepData, DoneData, ErrorData, TrainSnapshotData } from './types'
import { hydrateExperiment } from './lib/trainingState.js'

const LEGACY_STORAGE_KEY = 'arch_lab_experiments'

const modules = ref<ModulesInfo | null>(null)
const paramEstimate = ref<ParamEstimate>({
  param_count: 0,
  embedding_param_count: 0,
  lm_head_param_count: 0,
  remaining_param_count: 0,
  share_embedding_head: false,
})
const training = ref(false)
const stopping = ref(false)
const trainingProgress = ref<TrainingProgress | null>(null)
const gpuInfo = ref<GpuInfo | null>(null)
let trainingModelConfigSnapshot: ModelConfig | null = null
let trainingTrainConfigSnapshot: TrainConfig | null = null
let trainingParamCountSnapshot: number | null = null
let trainingRunNameSnapshot: string | null = null
let activeTrainingRunId: string | null = null
let trainWs: WebSocket | null = null
let trainWsConnectPromise: Promise<WebSocket | null> | null = null
let trainReconnectTimer: number | null = null
let shouldReconnectTrainWs = true
const handledCompletedRunIds = new Set<string>()
const handledErrorRunIds = new Set<string>()
const gpuLoading = ref(false)
const gpuError = ref<string | null>(null)
const currentLoss = ref<Array<{ step: number; loss: number; lr: number }>>([])
const displayedLoss = ref<Array<{ step: number; loss: number; lr: number }>>([])
const pendingExperimentName = ref('')
const savedExperiments = ref<Experiment[]>([])
const selectedExpIds = ref<string[]>([])
const detailExperimentId = ref<string | null>(null)
const detailExperimentOpen = ref(false)
const selectedExperiments = computed(() => savedExperiments.value.filter((exp) => selectedExpIds.value.includes(exp.id)))
const detailExperiment = computed(() => savedExperiments.value.find((exp) => exp.id === detailExperimentId.value) ?? null)

const defaultLayer = () => ({ attention_type: 'standard', attention_params: {}, ffn_type: 'swiglu', ffn_params: {} })

const modelConfig = ref<ModelConfig>({
  hidden_size: 256,
  vocab_size: 3204,
  max_seq_len: 64,
  rms_norm_eps: 1e-6,
  share_embedding_head: false,
  layers: Array.from({ length: 4 }, defaultLayer),
})

const trainConfig = ref<TrainConfig>({
  batch_size: 16,
  learning_rate: 3e-4,
  max_steps: null,
  warmup_steps: 25,
  data_order_seed: 0,
  model_init_seed: 0,
})

let gpuPollTimer: number | null = null

onMounted(async () => {
  try {
    const defaults = await api.fetchDefaults()
    modelConfig.value = {
      ...modelConfig.value,
      vocab_size: defaults.model.vocab_size,
    }
  } catch (error) {
    alert(`Failed to load default model config: ${error instanceof Error ? error.message : String(error)}`)
  }

  try {
    modules.value = await api.fetchModules()
  } catch (error) {
    alert(`Failed to load module metadata: ${error instanceof Error ? error.message : String(error)}`)
  }

  try {
    savedExperiments.value = await loadPersistedExperiments()
  } catch (error) {
    savedExperiments.value = loadLegacyExperiments()
    alert(`Failed to load persisted experiments: ${error instanceof Error ? error.message : String(error)}`)
  }

  if (savedExperiments.value.length > 0) {
    const latest = savedExperiments.value[0]
    selectedExpIds.value = [latest.id]
    detailExperimentId.value = latest.id
  }

  connectTrainStream()
  refreshGpuInfo()
  gpuPollTimer = window.setInterval(refreshGpuInfo, 2000)
})

onUnmounted(() => {
  shouldReconnectTrainWs = false
  if (trainReconnectTimer != null) {
    window.clearTimeout(trainReconnectTimer)
    trainReconnectTimer = null
  }
  trainWs?.close()
  trainWs = null
  if (gpuPollTimer != null) {
    window.clearInterval(gpuPollTimer)
    gpuPollTimer = null
  }
})

watch(modelConfig, async (cfg) => {
  try {
    paramEstimate.value = await api.estimateParams(cfg)
  } catch {
    // Ignore transient estimate failures while editing.
  }
}, { deep: true, immediate: true })

function updateLayers(layers: ModelConfig['layers']) {
  modelConfig.value = { ...modelConfig.value, layers }
}

async function refreshGpuInfo() {
  gpuLoading.value = true
  try {
    gpuInfo.value = await api.fetchGpuInfo()
    gpuError.value = null
  } catch (error) {
    gpuError.value = `GPU monitor unavailable: ${error instanceof Error ? error.message : String(error)}`
  } finally {
    gpuLoading.value = false
  }
}

function connectTrainStream(): Promise<WebSocket | null> {
  if (trainWs?.readyState === WebSocket.OPEN) {
    return Promise.resolve(trainWs)
  }
  if (trainWsConnectPromise) {
    return trainWsConnectPromise
  }

  shouldReconnectTrainWs = true
  if (trainReconnectTimer != null) {
    window.clearTimeout(trainReconnectTimer)
    trainReconnectTimer = null
  }

  trainWsConnectPromise = api.connectTrainWebSocket(handleTrainMessage, {
    onClose: () => {
      trainWs = null
      scheduleTrainReconnect()
    },
  })
    .then((ws) => {
      trainWs = ws
      return ws
    })
    .catch(() => {
      trainWs = null
      scheduleTrainReconnect()
      return null
    })
    .finally(() => {
      trainWsConnectPromise = null
    })

  return trainWsConnectPromise
}

function scheduleTrainReconnect() {
  if (!shouldReconnectTrainWs || trainReconnectTimer != null) {
    return
  }

  trainReconnectTimer = window.setTimeout(() => {
    trainReconnectTimer = null
    connectTrainStream()
  }, 1000)
}

function handleTrainMessage(msg: TrainMessage) {
  if (msg.type === 'snapshot') {
    applyTrainingSnapshot(msg)
    return
  }

  if (msg.type === 'step') {
    handleTrainingStep(msg)
    return
  }

  if (msg.type === 'done') {
    void handleTrainingDone(msg)
    return
  }

  handleTrainingError(msg)
}

function applyTrainingSnapshot(snapshot: TrainSnapshotData) {
  activeTrainingRunId = snapshot.run_id
  trainingModelConfigSnapshot = snapshot.model_config ? JSON.parse(JSON.stringify(snapshot.model_config)) : trainingModelConfigSnapshot
  trainingTrainConfigSnapshot = snapshot.train_config ? { ...snapshot.train_config } : trainingTrainConfigSnapshot
  trainingParamCountSnapshot = snapshot.param_count ?? trainingParamCountSnapshot
  trainingRunNameSnapshot = snapshot.run_name ?? trainingRunNameSnapshot

  currentLoss.value = snapshot.loss_history.map((point) => ({
    step: point.step,
    loss: point.loss,
    lr: point.lr,
  }))
  displayedLoss.value = [...currentLoss.value]

  if (snapshot.progress) {
    trainingProgress.value = toTrainingProgress(snapshot.progress)
  } else if (snapshot.status !== 'done') {
    trainingProgress.value = null
  }

  training.value = snapshot.training || snapshot.status === 'running' || snapshot.status === 'stopping'
  stopping.value = snapshot.status === 'stopping'

  if (snapshot.status === 'done' && snapshot.done) {
    void handleTrainingDone(snapshot.done, snapshot)
    return
  }

  if (snapshot.status === 'error' && snapshot.error) {
    handleTrainingError(snapshot.error)
    return
  }

  if (snapshot.status === 'idle') {
    resetTrainingSnapshots()
  }
}

function handleTrainingStep(msg: StepData) {
  activeTrainingRunId = msg.run_id ?? activeTrainingRunId
  training.value = true
  appendLossPoint({ step: msg.step, loss: msg.loss, lr: msg.lr })
  trainingProgress.value = toTrainingProgress(msg)
}

async function handleTrainingDone(msg: DoneData, snapshot: TrainSnapshotData | null = null) {
  const runId = msg.run_id ?? snapshot?.run_id ?? activeTrainingRunId
  training.value = false
  stopping.value = false
  trainingProgress.value = null

  if (runId && handledCompletedRunIds.has(runId)) {
    return
  }
  if (runId) {
    handledCompletedRunIds.add(runId)
  }

  if (snapshot?.loss_history) {
    currentLoss.value = snapshot.loss_history.map((point) => ({
      step: point.step,
      loss: point.loss,
      lr: point.lr,
    }))
    displayedLoss.value = [...currentLoss.value]
  }

  const defaultName = `Exp ${savedExperiments.value.length + 1}`
  const experimentName = (snapshot?.run_name ?? trainingRunNameSnapshot ?? pendingExperimentName.value).trim() || defaultName
  const colorIndex = getNextExperimentColorIndex()
  const modelSnapshot = snapshot?.model_config ?? trainingModelConfigSnapshot ?? JSON.parse(JSON.stringify(modelConfig.value))
  const trainSnapshot = snapshot?.train_config ?? trainingTrainConfigSnapshot ?? { ...trainConfig.value }
  const paramCountSnapshot = snapshot?.param_count ?? trainingParamCountSnapshot ?? paramEstimate.value.param_count

  const experimentId = runId ?? Date.now().toString()
  const completedAt = snapshot?.updated_at != null ? Math.round(snapshot.updated_at * 1000) : Date.now()

  const localExperiment = hydrateExperiment({
    id: experimentId,
    name: experimentName,
    colorIndex,
    timestamp: completedAt,
    model_config: JSON.parse(JSON.stringify(modelSnapshot)),
    train_config: { ...trainSnapshot },
    param_count: paramCountSnapshot,
    loss_history: [...currentLoss.value],
    final_loss: msg.final_loss,
    completed_steps: msg.total_steps,
    target_total_steps: msg.target_total_steps,
    dataset_total_steps: msg.dataset_total_steps,
    elapsed_seconds: msg.elapsed_seconds,
  }, colorIndex)

  const persisted = await persistCompletedExperiment(localExperiment)
  if (!persisted) {
    if (runId) {
      handledCompletedRunIds.delete(runId)
    }
    return
  }

  if (runId) {
    try {
      await api.clearFinishedTraining(runId)
    } catch {
      // Keep the local result; the in-memory handled set prevents duplicate saves in this page.
    }
  }
  resetTrainingSnapshots()
}

function handleTrainingError(msg: ErrorData) {
  const runId = msg.run_id ?? activeTrainingRunId
  training.value = false
  stopping.value = false
  trainingProgress.value = null
  if (runId && handledErrorRunIds.has(runId)) {
    return
  }
  if (runId) {
    handledErrorRunIds.add(runId)
  }
  alert(`Training error: ${msg.message}`)
  if (runId) {
    void api.clearFinishedTraining(runId).finally(() => {
      if (activeTrainingRunId === runId) {
        resetTrainingSnapshots()
      }
    })
  }
}

function appendLossPoint(point: { step: number; loss: number; lr: number }) {
  const existingIndex = currentLoss.value.findIndex((item) => item.step === point.step)
  if (existingIndex >= 0) {
    currentLoss.value.splice(existingIndex, 1, point)
    displayedLoss.value = [...currentLoss.value]
    return
  }

  const lastPoint = currentLoss.value.at(-1)
  if (lastPoint && point.step < lastPoint.step) {
    currentLoss.value = [...currentLoss.value, point].sort((a, b) => a.step - b.step)
    displayedLoss.value = [...currentLoss.value]
    return
  }

  currentLoss.value.push(point)
  displayedLoss.value.push(point)
}

function toTrainingProgress(msg: StepData): TrainingProgress {
  return {
    step: msg.step,
    targetTotalSteps: msg.target_total_steps,
    datasetTotalSteps: msg.dataset_total_steps,
    progressPct: msg.progress_pct,
    elapsedSeconds: msg.elapsed_seconds,
    etaSeconds: msg.eta_seconds,
    loss: msg.loss,
    lr: msg.lr,
    tokensPerSec: msg.tokens_per_sec,
  }
}

function resetTrainingSnapshots() {
  activeTrainingRunId = null
  trainingModelConfigSnapshot = null
  trainingTrainConfigSnapshot = null
  trainingParamCountSnapshot = null
  trainingRunNameSnapshot = null
}

async function startTrain() {
  currentLoss.value = []
  displayedLoss.value = []
  trainingProgress.value = null
  stopping.value = false
  training.value = true
  trainingModelConfigSnapshot = JSON.parse(JSON.stringify(modelConfig.value))
  trainingTrainConfigSnapshot = { ...trainConfig.value }
  trainingParamCountSnapshot = paramEstimate.value.param_count
  trainingRunNameSnapshot = pendingExperimentName.value
  activeTrainingRunId = null

  try {
    await connectTrainStream()
    const started = await api.startTraining(modelConfig.value, trainConfig.value, {
      runName: trainingRunNameSnapshot ?? undefined,
      paramCount: trainingParamCountSnapshot ?? undefined,
    })
    activeTrainingRunId = started.run_id
  } catch (error) {
    training.value = false
    stopping.value = false
    trainingProgress.value = null
    resetTrainingSnapshots()
    alert(`Training error: ${error instanceof Error ? error.message : String(error)}`)
  }
}

async function persistCompletedExperiment(experiment: Experiment): Promise<boolean> {
  let persisted: Experiment
  try {
    persisted = hydrateExperiment(await api.saveExperiment(experiment), experiment.colorIndex ?? 0)
  } catch (error) {
    alert(`Experiment was not persisted to disk: ${error instanceof Error ? error.message : String(error)}`)
    return false
  }

  savedExperiments.value = upsertExperiment(savedExperiments.value, persisted)
  pendingExperimentName.value = ''
  currentLoss.value = []
  displayedLoss.value = []
  onSelectExperiments([...new Set([...selectedExpIds.value, persisted.id])])
  detailExperimentId.value = persisted.id
  detailExperimentOpen.value = false
  return true
}

async function stopTrain() {
  if (stopping.value) {
    return
  }

  stopping.value = true
  try {
    await api.stopTraining()
  } catch (error) {
    stopping.value = false
    alert(`Stop training failed: ${error instanceof Error ? error.message : String(error)}`)
  }
}

function onSelectExperiments(ids: string[]) {
  selectedExpIds.value = ids
}

function openExperimentDetails(id: string) {
  detailExperimentId.value = id
  detailExperimentOpen.value = true
}

function closeExperimentDetails() {
  detailExperimentOpen.value = false
}

async function renameExperiment(id: string, name: string) {
  try {
    await api.updateExperimentName(id, name)
  } catch (error) {
    alert(`Failed to rename experiment: ${error instanceof Error ? error.message : String(error)}`)
    return
  }
  const exp = savedExperiments.value.find((e) => e.id === id)
  if (exp) {
    exp.name = name
  }
}

async function deleteExperiment(id: string) {
  try {
    await api.deleteExperiment(id)
  } catch (error) {
    alert(`Failed to delete experiment: ${error instanceof Error ? error.message : String(error)}`)
    return
  }

  savedExperiments.value = savedExperiments.value.filter((exp) => exp.id !== id)
  if (selectedExpIds.value.includes(id)) {
    onSelectExperiments(selectedExpIds.value.filter((value) => value !== id))
  }
  if (detailExperimentId.value === id) {
    detailExperimentId.value = savedExperiments.value[0]?.id ?? null
    detailExperimentOpen.value = detailExperimentId.value != null && detailExperimentOpen.value
  }
}

async function deleteSelectedExperiments(ids: string[]) {
  const uniqueIds = [...new Set(ids)]
  if (uniqueIds.length === 0) {
    return
  }

  const failures: string[] = []
  for (const id of uniqueIds) {
    try {
      await api.deleteExperiment(id)
    } catch {
      failures.push(id)
    }
  }

  if (failures.length === uniqueIds.length) {
    alert('Failed to delete selected experiments.')
    return
  }

  const failedSet = new Set(failures)
  const deletedSet = new Set(uniqueIds.filter((id) => !failedSet.has(id)))
  savedExperiments.value = savedExperiments.value.filter((exp) => !deletedSet.has(exp.id))
  onSelectExperiments(selectedExpIds.value.filter((id) => !deletedSet.has(id)))

  if (detailExperimentId.value && deletedSet.has(detailExperimentId.value)) {
    detailExperimentId.value = savedExperiments.value[0]?.id ?? null
    detailExperimentOpen.value = detailExperimentId.value != null && detailExperimentOpen.value
  }

  if (failures.length > 0) {
    alert(`Deleted ${uniqueIds.length - failures.length} experiment(s), but ${failures.length} failed.`)
  }
}

function getNextExperimentColorIndex(): number {
  return savedExperiments.value.reduce((maxIndex, experiment, index) => {
    const colorIndex = experiment.colorIndex ?? index
    return Math.max(maxIndex, colorIndex)
  }, -1) + 1
}

async function loadPersistedExperiments(): Promise<Experiment[]> {
  const persisted = (await api.fetchExperiments()).map((experiment, index) => hydrateExperiment(experiment, index))
  if (persisted.length > 0) {
    clearLegacyExperiments()
    return persisted
  }

  const legacy = loadLegacyExperiments()
  if (legacy.length === 0) {
    return []
  }

  const migrated: Experiment[] = []
  for (const experiment of legacy) {
    const fallbackColorIndex = experiment.colorIndex ?? migrated.length
    try {
      const saved = await api.saveExperiment(experiment)
      migrated.push(hydrateExperiment(saved, fallbackColorIndex))
    } catch {
      migrated.push(hydrateExperiment(experiment, fallbackColorIndex))
    }
  }

  clearLegacyExperiments()
  return migrated.sort((a, b) => b.timestamp - a.timestamp)
}

function loadLegacyExperiments(): Experiment[] {
  try {
    const raw = localStorage.getItem(LEGACY_STORAGE_KEY)
    if (!raw) {
      return []
    }
    const parsed = JSON.parse(raw) as Experiment[]
    return parsed.map((experiment, index) => hydrateExperiment(experiment, index))
  } catch {
    return []
  }
}

function clearLegacyExperiments() {
  localStorage.removeItem(LEGACY_STORAGE_KEY)
}

function upsertExperiment(experiments: Experiment[], nextExperiment: Experiment): Experiment[] {
  const byId = new Map(experiments.map((experiment) => [experiment.id, experiment]))
  byId.set(nextExperiment.id, nextExperiment)
  return [...byId.values()].sort((a, b) => b.timestamp - a.timestamp)
}
</script>
