<template>
  <div class="section">
    <h3>Model Parameters</h3>
    <div class="param-count">
      {{ formatNumber(estimate.param_count) }}
      <small>trainable parameters</small>
    </div>
    <div class="param-breakdown">
      <div class="param-row">
        <span>Embedding</span>
        <strong>{{ formatNumber(estimate.embedding_param_count) }}</strong>
      </div>
      <div class="param-row">
        <span>LM Head</span>
        <strong>
          {{ estimate.share_embedding_head ? 'Shared' : formatNumber(estimate.lm_head_param_count) }}
        </strong>
      </div>
      <div class="param-row">
        <span>Remaining Params</span>
        <strong>{{ formatNumber(estimate.remaining_param_count) }}</strong>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { ParamEstimate } from '../types'

defineProps<{ estimate: ParamEstimate }>()

function formatNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M'
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K'
  return n.toString()
}
</script>
