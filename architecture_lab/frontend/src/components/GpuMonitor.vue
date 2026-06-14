<template>
  <div class="section gpu-panel">
    <div class="gpu-header">
      <h3>GPU Monitor</h3>
      <span class="gpu-source">{{ statusLabel }}</span>
    </div>

    <div v-if="loading && !info" class="gpu-empty">Detecting GPU...</div>
    <div v-else-if="error" class="gpu-empty">{{ error }}</div>
    <div v-else-if="!info?.available" class="gpu-empty">{{ info?.message ?? 'No GPU detected.' }}</div>

    <div v-else class="gpu-list">
      <div v-for="gpu in info.gpus" :key="gpu.index" class="gpu-device">
        <div class="gpu-device-top">
          <strong>GPU {{ gpu.index }}</strong>
          <span>{{ gpu.name }}</span>
        </div>

        <div class="gpu-meter-row">
          <span>VRAM</span>
          <strong>{{ formatMemory(gpu.memory_used_mb, gpu.memory_total_mb) }}</strong>
        </div>
        <div class="gpu-meter">
          <div class="gpu-meter-fill" :style="{ width: `${clampPercent(gpu.memory_usage_pct)}%` }"></div>
        </div>

        <div class="gpu-stats-grid">
          <span>Util {{ formatPercent(gpu.utilization_pct) }}</span>
          <span>Temp {{ formatTemperature(gpu.temperature_c) }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { GpuInfo } from '../types'

const props = defineProps<{
  info: GpuInfo | null
  loading: boolean
  error: string | null
}>()

const statusLabel = computed(() => {
  if (props.loading && !props.info) return 'checking'
  if (props.error) return 'error'
  return props.info?.source ?? 'none'
})

function clampPercent(value: number | null): number {
  if (value == null || Number.isNaN(value)) return 0
  return Math.min(100, Math.max(0, value))
}

function formatMemory(used: number | null, total: number | null): string {
  if (used == null || total == null) return '--'
  return `${formatMb(used)} / ${formatMb(total)}`
}

function formatMb(value: number): string {
  if (value >= 1024) return `${(value / 1024).toFixed(1)} GB`
  return `${value} MB`
}

function formatPercent(value: number | null): string {
  return value == null ? '--' : `${value}%`
}

function formatTemperature(value: number | null): string {
  return value == null ? '--' : `${value} C`
}
</script>
