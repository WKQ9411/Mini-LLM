<template>
  <div class="section">
    <h3>Layer Configuration</h3>
    <div v-for="(layer, i) in layers" :key="i" class="layer-card">
      <div class="layer-card-header">
        <h4>Layer {{ i }}</h4>
        <button type="button" class="layer-action-button" @click="toggleApplyPanel(i)">
          Apply To
        </button>
      </div>

      <div v-if="applyPanelIndex === i" class="apply-panel">
        <div class="apply-mode-row">
          <button
            type="button"
            class="segmented-button"
            :class="{ active: applyMode === 'all' }"
            @click="setApplyMode('all')"
          >
            All Layers
          </button>
          <button
            type="button"
            class="segmented-button"
            :class="{ active: applyMode === 'specific' }"
            @click="setApplyMode('specific')"
          >
            Specific Layers
          </button>
        </div>

        <label v-if="applyMode === 'specific'" class="apply-input-row">
          <span>Layer IDs</span>
          <input v-model="applyTargetInput" type="text" placeholder="0,2,4" />
        </label>

        <div class="apply-help">
          Copies this layer's full attention and FFN configuration. Layer IDs use the current 0-based numbering.
        </div>
        <div v-if="applyError" class="apply-error">{{ applyError }}</div>

        <div class="apply-actions">
          <button type="button" class="primary" @click="applyCurrentLayer(i)">Apply</button>
          <button type="button" @click="closeApplyPanel">Cancel</button>
        </div>
      </div>

      <label><span>Attention</span>
        <select :value="layer.attention_type" @change="updateLayer(i, 'attention_type', ($event.target as HTMLSelectElement).value)">
          <option v-for="(info, key) in modules?.attention_types" :key="key" :value="key">{{ info.name }}</option>
        </select>
      </label>
      <div v-for="(p, pk) in currentAttnParams(i)" :key="pk" class="param-group">
        <label v-if="p.type === 'select'">
          <span>{{ pk }}</span>
          <select
            :value="String(getParamValue(layer, 'attention_params', pk, p))"
            @change="updateParam(i, 'attention_params', pk, ($event.target as HTMLSelectElement).value)"
            :disabled="isParamLocked(layer, 'attention_params', pk)"
            :class="{ locked: isParamLocked(layer, 'attention_params', pk) }"
          >
            <option v-for="option in p.options" :key="option" :value="option">{{ option }}</option>
          </select>
        </label>
        <label v-else>
          <span>{{ pk }}</span>
          <input
            type="number"
            :value="Number(getParamValue(layer, 'attention_params', pk, p))"
            @input="updateParam(i, 'attention_params', pk, Number(($event.target as HTMLInputElement).value))"
            :min="p.min"
            :max="p.max"
            :step="p.type === 'float' ? 'any' : '1'"
            :disabled="isParamLocked(layer, 'attention_params', pk)"
            :class="{ locked: isParamLocked(layer, 'attention_params', pk) }"
          />
        </label>
      </div>
      <label><span>FFN</span>
        <select :value="layer.ffn_type" @change="updateLayer(i, 'ffn_type', ($event.target as HTMLSelectElement).value)">
          <option v-for="(info, key) in modules?.ffn_types" :key="key" :value="key">{{ info.name }}</option>
        </select>
      </label>
      <div v-for="(p, pk) in currentFfnParams(i)" :key="pk" class="param-group">
        <label v-if="p.type === 'select'">
          <span>{{ pk }}</span>
          <select
            :value="String(getParamValue(layer, 'ffn_params', pk, p))"
            @change="updateParam(i, 'ffn_params', pk, ($event.target as HTMLSelectElement).value)"
            :disabled="isParamLocked(layer, 'ffn_params', pk)"
            :class="{ locked: isParamLocked(layer, 'ffn_params', pk) }"
          >
            <option v-for="option in p.options" :key="option" :value="option">{{ option }}</option>
          </select>
        </label>
        <label v-else>
          <span>{{ pk }}</span>
          <input
            type="number"
            :value="Number(getParamValue(layer, 'ffn_params', pk, p))"
            @input="updateParam(i, 'ffn_params', pk, Number(($event.target as HTMLInputElement).value))"
            :min="p.min"
            :max="p.max"
            :step="p.type === 'float' ? 'any' : '1'"
            :disabled="isParamLocked(layer, 'ffn_params', pk)"
            :class="{ locked: isParamLocked(layer, 'ffn_params', pk) }"
          />
        </label>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { LayerConfig, ModulesInfo } from '../types'
import {
  applyLayerConfigToTargets,
  getParamValue,
  isParamLocked,
  parseTargetLayerIds,
  updateLayerParamValue,
} from './layerConfigState.js'

const props = defineProps<{ layers: LayerConfig[]; modules: ModulesInfo | null }>()
const emit = defineEmits<{ 'update:layers': [value: LayerConfig[]] }>()

const applyPanelIndex = ref<number | null>(null)
const applyMode = ref<'all' | 'specific'>('all')
const applyTargetInput = ref('')
const applyError = ref('')

function currentAttnParams(i: number) {
  const t = props.layers[i]?.attention_type
  return (t && props.modules?.attention_types[t]?.params) || {}
}

function currentFfnParams(i: number) {
  const t = props.layers[i]?.ffn_type
  return (t && props.modules?.ffn_types[t]?.params) || {}
}

function updateLayer(i: number, field: 'attention_type' | 'ffn_type', value: string) {
  const layers = props.layers.map((l, idx) => {
    if (idx !== i) return l
    if (field === 'attention_type') {
      return { ...l, attention_type: value, attention_params: {} }
    }
    return { ...l, ffn_type: value, ffn_params: {} }
  })
  emit('update:layers', layers)
}

function updateParam(i: number, group: 'attention_params' | 'ffn_params', key: string, value: number | string) {
  const layers = props.layers.map((l, idx) =>
    idx === i ? updateLayerParamValue(l, group, key, value) : l
  )
  emit('update:layers', layers)
}

function toggleApplyPanel(index: number) {
  if (applyPanelIndex.value === index) {
    closeApplyPanel()
    return
  }

  applyPanelIndex.value = index
  applyMode.value = 'all'
  applyTargetInput.value = ''
  applyError.value = ''
}

function closeApplyPanel() {
  applyPanelIndex.value = null
  applyMode.value = 'all'
  applyTargetInput.value = ''
  applyError.value = ''
}

function setApplyMode(mode: 'all' | 'specific') {
  applyMode.value = mode
  applyError.value = ''
}

function applyCurrentLayer(sourceIndex: number) {
  const targetIndices = (
    applyMode.value === 'all'
      ? props.layers.map((_, index) => index).filter((index) => index !== sourceIndex)
      : parseTargetLayerIds(applyTargetInput.value, props.layers.length, sourceIndex)
  )

  if (targetIndices.length === 0) {
    applyError.value = (
      applyMode.value === 'all'
        ? 'There are no other layers to apply this configuration to.'
        : 'Enter at least one valid target layer ID.'
    )
    return
  }

  emit('update:layers', applyLayerConfigToTargets(props.layers, sourceIndex, targetIndices))
  closeApplyPanel()
}
</script>
