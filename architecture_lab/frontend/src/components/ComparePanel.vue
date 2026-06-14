<template>
  <div class="section history-panel">
    <div class="section-header">
      <div class="section-header-group">
        <h3>Experiment History</h3>
        <span class="section-meta">{{ allExperiments.length }} runs</span>
      </div>
      <div class="section-actions">
        <button type="button" :disabled="allExperiments.length === 0" @click="selectAll">
          Select All
        </button>
        <button type="button" :disabled="selected.length === 0" @click="clearSelection">
          Clear
        </button>
        <button type="button" class="danger" :disabled="selected.length === 0" @click="deleteSelected">
          Delete Selected
        </button>
      </div>
    </div>

    <div v-if="allExperiments.length === 0" class="history-empty">
      No experiments saved yet.
    </div>

    <div v-else class="history-list">
      <div
        v-for="exp in allExperiments"
        :key="exp.id"
        class="experiment-item"
        :class="{
          selected: selected.includes(exp.id),
          inspected: detailId === exp.id,
        }"
      >
        <label class="experiment-select">
          <input type="checkbox" :value="exp.id" v-model="selected" />
          <span class="swatch" :style="{ backgroundColor: resolveColor(exp.colorIndex ?? 0) }"></span>
        </label>

        <div class="experiment-main">
          <div class="experiment-topline">
            <input
              v-if="editingId === exp.id"
              ref="editInput"
              v-model="editName"
              class="name name-input"
              @keydown.enter="commitRename(exp.id)"
              @keydown.escape="cancelRename"
              @blur="commitRename(exp.id)"
            />
            <span v-else class="name" @click="startRename(exp)" title="Click to rename">{{ exp.name }}</span>
            <span class="meta">Loss {{ exp.final_loss?.toFixed(3) ?? '--' }}</span>
          </div>
          <div class="experiment-subline">
            <span>{{ formatNumber(exp.param_count) }} params</span>
            <span>{{ exp.model_config.layers.length }} layers</span>
            <span>{{ formatTimestamp(exp.timestamp) }}</span>
          </div>
          <div class="experiment-arch">
            {{ summarizeLayerStack(exp.model_config.layers) }}
          </div>
        </div>

        <div class="experiment-actions">
          <button type="button" @click.stop="emit('detail', exp.id)">Detail</button>
          <button type="button" @click.stop="emit('delete', exp.id)">Delete</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import type { Experiment } from '../types'
import { summarizeLayerStack } from '../lib/trainingState.js'

const props = defineProps<{
  experiments: Experiment[]
  selectedIds: string[]
  detailId: string | null
}>()

const emit = defineEmits<{
  'select': [ids: string[]]
  'delete': [id: string]
  'delete-selected': [ids: string[]]
  'detail': [id: string]
  'rename': [id: string, name: string]
}>()

const selected = ref<string[]>([])
const allExperiments = ref<Experiment[]>([])
const editingId = ref<string | null>(null)
const editName = ref('')
const editInput = ref<HTMLInputElement | null>(null)

watch(() => props.experiments, (value) => {
  allExperiments.value = [...value].sort((a, b) => b.timestamp - a.timestamp)
}, { immediate: true })

watch(() => props.selectedIds, (value) => {
  if (isSameSelection(selected.value, value)) {
    return
  }
  selected.value = [...value]
}, { immediate: true })

watch(selected, (value) => {
  if (isSameSelection(value, props.selectedIds)) {
    return
  }
  emit('select', [...value])
})

function isSameSelection(a: string[], b: string[]) {
  return a.length === b.length && a.every((value, index) => value === b[index])
}

function selectAll() {
  emit('select', allExperiments.value.map((exp) => exp.id))
}

function clearSelection() {
  emit('select', [])
}

function deleteSelected() {
  emit('delete-selected', [...selected.value])
}

function startRename(exp: Experiment) {
  editingId.value = exp.id
  editName.value = exp.name
  nextTick(() => editInput.value?.focus())
}

function commitRename(id: string) {
  const trimmed = editName.value.trim()
  if (editingId.value !== id) return // already committed or cancelled
  editingId.value = null
  if (!trimmed) return
  const original = allExperiments.value.find((e) => e.id === id)
  if (original && trimmed !== original.name) {
    emit('rename', id, trimmed)
  }
}

function cancelRename() {
  editingId.value = null
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toString()
}

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp).toLocaleString([], {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function resolveColor(colorIndex: number): string {
  const palette = [
    '#4a9eff', '#f6c344', '#5ad8a6', '#ff7a90', '#a78bfa', '#ff9f43', '#7dd3fc', '#f472b6',
    '#34d399', '#f87171', '#60a5fa', '#fbbf24', '#22d3ee', '#c084fc', '#fb7185', '#a3e635',
  ]
  return palette[colorIndex % palette.length]
}
</script>
