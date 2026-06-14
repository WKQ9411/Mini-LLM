<template>
  <div class="section">
    <h3>Training</h3>
    <div v-if="training && progress" class="training-status">
      <div class="training-status-top">
        <strong>{{ stopping ? 'Stopping...' : `${progress.progressPct.toFixed(1)}%` }}</strong>
        <span>{{ progress.step }} / {{ progress.targetTotalSteps }} steps</span>
        <span>{{ stopping ? 'Stop requested' : `ETA ${formatDuration(progress.etaSeconds)}` }}</span>
      </div>
      <div class="training-progress-bar">
        <div class="training-progress-fill" :style="{ width: `${Math.min(progress.progressPct, 100)}%` }"></div>
      </div>
      <div class="training-status-grid">
        <span>Loss {{ progress.loss.toFixed(4) }}</span>
        <span>LR {{ formatLearningRate(progress.lr) }}</span>
        <span>{{ progress.tokensPerSec.toFixed(0) }} tok/s</span>
        <span>Elapsed {{ formatDuration(progress.elapsedSeconds) }}</span>
      </div>
    </div>
    <label><span>Run Name</span><input type="text" v-model="runName" placeholder="Exp 1" /></label>
    <label><span>Batch Size</span><input type="number" v-model.number="config.batch_size" min="1" max="32" /></label>
    <label><span>Learning Rate</span><input type="number" v-model.number="config.learning_rate" min="1e-5" max="1e-2" step="1e-5" /></label>
    <label><span>Max Steps</span><input type="number" :value="config.max_steps" @input="updateMaxSteps" placeholder="auto (1 epoch)" min="1" step="100" /></label>
    <label><span>Warmup Steps</span><input type="number" v-model.number="config.warmup_steps" min="0" max="500" /></label>
    <label><span>Data Order Seed</span><input type="number" v-model.number="config.data_order_seed" min="0" step="1" /></label>
    <label><span>Model Init Seed</span><input type="number" v-model.number="config.model_init_seed" min="0" step="1" /></label>
    <div class="btn-row" style="margin-top: 10px;">
      <button class="primary" @click="$emit('start')" :disabled="training || stopping">Start Train</button>
      <button class="danger" @click="$emit('stop')" :disabled="!training || stopping">{{ stopping ? 'Stopping...' : 'Stop' }}</button>
    </div>
  </div>
</template>

<script setup lang="ts">
 import type { TrainConfig, TrainingProgress } from '../types'
 import { formatDuration } from '../lib/trainingState.js'

defineProps<{ training: boolean; stopping: boolean; progress: TrainingProgress | null }>()
const config = defineModel<TrainConfig>('config', { required: true })
const runName = defineModel<string>('runName', { required: true })
defineEmits<{ start: []; stop: [] }>()

function updateMaxSteps(event: Event) {
  const val = (event.target as HTMLInputElement).value
  config.value = { ...config.value, max_steps: val === '' ? null : Number(val) }
}

function formatLearningRate(lr: number): string {
  return lr >= 0.001 ? lr.toFixed(4) : lr.toExponential(2)
}
</script>
