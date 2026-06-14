<template>
  <Teleport to="body">
    <div v-if="open" class="modal-overlay" @click.self="emit('close')">
      <div class="modal-card detail-modal">
        <div class="detail-header">
          <div>
            <div class="detail-kicker detail-kicker-brand">
              <img src="/logo.png" alt="Architecture Lab logo" class="detail-kicker-logo" />
              <span>Experiment Details</span>
            </div>
            <div class="detail-title">{{ experiment?.name ?? 'Unknown Experiment' }}</div>
            <div v-if="experiment" class="detail-subtitle">
              {{ overview.architectureSummary }}
            </div>
          </div>
          <div class="detail-header-actions">
            <div v-if="experiment" class="detail-timestamp">{{ formatTimestamp(experiment.timestamp) }}</div>
            <button type="button" @click="emit('close')">Close</button>
          </div>
        </div>

        <div v-if="!experiment" class="detail-empty">
          This experiment is no longer available.
        </div>

        <template v-else>
          <div class="detail-grid">
            <div class="detail-block">
              <h4>Training Summary</h4>
              <div class="detail-row"><span>Final Loss</span><strong>{{ experiment.final_loss?.toFixed(4) ?? '--' }}</strong></div>
              <div class="detail-row"><span>Completed Steps</span><strong>{{ experiment.completed_steps }} / {{ experiment.target_total_steps }}</strong></div>
              <div class="detail-row"><span>Dataset Steps</span><strong>{{ experiment.dataset_total_steps ?? '--' }}</strong></div>
              <div class="detail-row"><span>Elapsed</span><strong>{{ overview.formattedElapsed }}</strong></div>
              <div class="detail-row"><span>Batch / Seq</span><strong>{{ experiment.train_config.batch_size }} / {{ experiment.model_config.max_seq_len }}</strong></div>
              <div class="detail-row"><span>Learning Rate</span><strong>{{ formatLearningRate(experiment.train_config.learning_rate) }}</strong></div>
            </div>

            <div class="detail-block">
              <h4>Model Summary</h4>
              <div class="detail-row"><span>Parameters</span><strong>{{ formatNumber(experiment.param_count) }}</strong></div>
              <div class="detail-row"><span>Layers</span><strong>{{ overview.layerCount }}</strong></div>
              <div class="detail-row"><span>Hidden Size</span><strong>{{ experiment.model_config.hidden_size }}</strong></div>
              <div class="detail-row"><span>Vocab Size</span><strong>{{ experiment.model_config.vocab_size }}</strong></div>
              <div class="detail-row"><span>Max Seq Len</span><strong>{{ experiment.model_config.max_seq_len }}</strong></div>
              <div class="detail-row"><span>Share LM Head</span><strong>{{ experiment.model_config.share_embedding_head ? 'Yes' : 'No' }}</strong></div>
            </div>
          </div>

          <div class="detail-block">
            <h4>Layer Stack</h4>
            <details
              v-for="(layer, index) in experiment.model_config.layers"
              :key="index"
              class="layer-detail"
            >
              <summary>
                <span>Layer {{ index }}</span>
                <span class="layer-detail-summary">{{ layer.attention_type }} / {{ layer.ffn_type }}</span>
              </summary>
              <div class="layer-detail-content">
                <div class="detail-subgrid">
                  <div>
                    <div class="detail-label">Attention Params</div>
                    <div v-if="Object.keys(layer.attention_params).length === 0" class="detail-empty-inline">Default</div>
                    <div v-for="(value, key) in layer.attention_params" :key="`attn-${key}`" class="detail-row">
                      <span>{{ key }}</span>
                      <strong>{{ value }}</strong>
                    </div>
                  </div>
                  <div>
                    <div class="detail-label">FFN Params</div>
                    <div v-if="Object.keys(layer.ffn_params).length === 0" class="detail-empty-inline">Default</div>
                    <div v-for="(value, key) in layer.ffn_params" :key="`ffn-${key}`" class="detail-row">
                      <span>{{ key }}</span>
                      <strong>{{ value }}</strong>
                    </div>
                  </div>
                </div>
              </div>
            </details>
          </div>
        </template>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Experiment } from '../types'
import { buildExperimentOverview } from '../lib/trainingState.js'

const props = defineProps<{
  open: boolean
  experiment: Experiment | null
}>()

const emit = defineEmits<{ close: [] }>()

const overview = computed(() => (
  props.experiment
    ? buildExperimentOverview(props.experiment)
    : {
        layerCount: 0,
        architectureSummary: '',
        formattedElapsed: '--',
        formattedEta: '--',
      }
))

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toString()
}

function formatLearningRate(lr: number): string {
  return lr >= 0.001 ? lr.toFixed(4) : lr.toExponential(2)
}

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp).toLocaleString()
}
</script>
