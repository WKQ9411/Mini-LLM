<template>
  <div class="section">
    <h3>Global Params</h3>
    <label><span>Hidden Size</span><input type="number" :value="model.hidden_size" @input="update('hidden_size', $event)" min="32" max="1024" step="32" /></label>
    <label><span>Vocab Size</span><input type="number" :value="model.vocab_size" @input="update('vocab_size', $event)" min="1000" max="100000" /></label>
    <label><span>Max Seq Len</span><input type="number" :value="model.max_seq_len" @input="update('max_seq_len', $event)" min="16" max="512" /></label>
    <label><span>Num Layers</span><input type="number" :value="model.layers.length" @input="updateLayers($event)" min="1" max="24" /></label>
    <label class="checkbox-row">
      <span>Share LM Head</span>
      <input type="checkbox" :checked="model.share_embedding_head" @change="updateShareEmbeddingHead($event)" />
    </label>
  </div>
</template>

<script setup lang="ts">
import type { ModelConfig } from '../types'

const props = defineProps<{ model: ModelConfig }>()
const emit = defineEmits<{ 'update:model': [value: ModelConfig] }>()

function update(key: string, event: Event) {
  const val = Number((event.target as HTMLInputElement).value)
  emit('update:model', { ...props.model, [key]: val })
}

function updateLayers(event: Event) {
  const n = Number((event.target as HTMLInputElement).value)
  const layers = [...props.model.layers]
  const defaultLayer = { attention_type: 'standard', attention_params: {}, ffn_type: 'swiglu', ffn_params: {} }
  while (layers.length < n) layers.push({ ...defaultLayer, attention_params: {}, ffn_params: {} })
  while (layers.length > n) layers.pop()
  emit('update:model', { ...props.model, layers })
}

function updateShareEmbeddingHead(event: Event) {
  emit('update:model', { ...props.model, share_embedding_head: (event.target as HTMLInputElement).checked })
}
</script>
