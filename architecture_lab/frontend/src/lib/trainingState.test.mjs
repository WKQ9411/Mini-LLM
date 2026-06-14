import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildExperimentOverview,
  formatDuration,
  hydrateExperiment,
  summarizeLayerStack,
} from './trainingState.js'

test('formatDuration formats compact time spans', () => {
  assert.equal(formatDuration(null), '--')
  assert.equal(formatDuration(9), '00:09')
  assert.equal(formatDuration(125), '02:05')
  assert.equal(formatDuration(3723), '01:02:03')
})

test('summarizeLayerStack compresses repeated layer patterns', () => {
  const summary = summarizeLayerStack([
    { attention_type: 'standard', attention_params: {}, ffn_type: 'swiglu', ffn_params: {} },
    { attention_type: 'standard', attention_params: {}, ffn_type: 'swiglu', ffn_params: {} },
    { attention_type: 'mla', attention_params: {}, ffn_type: 'geglu', ffn_params: {} },
    { attention_type: 'mla', attention_params: {}, ffn_type: 'geglu', ffn_params: {} },
    { attention_type: 'gated', attention_params: {}, ffn_type: 'moe', ffn_params: {} },
  ])

  assert.equal(summary, '2x standard / swiglu, 2x mla / geglu, 1x gated / moe')
})

test('buildExperimentOverview exposes compact experiment metadata', () => {
  const overview = buildExperimentOverview({
    id: '1',
    name: 'Exp 1',
    timestamp: 1000,
    model_config: {
      hidden_size: 128,
      vocab_size: 3204,
      max_seq_len: 64,
      rms_norm_eps: 1e-6,
      share_embedding_head: false,
      layers: [
        { attention_type: 'standard', attention_params: {}, ffn_type: 'swiglu', ffn_params: {} },
        { attention_type: 'standard', attention_params: {}, ffn_type: 'swiglu', ffn_params: {} },
      ],
    },
    train_config: {
      batch_size: 4,
      learning_rate: 3e-4,
      max_steps: 500,
      warmup_steps: 25,
      data_order_seed: 0,
      model_init_seed: 0,
    },
    param_count: 123456,
    loss_history: [{ step: 1, loss: 2.5, lr: 0.001 }],
    final_loss: 2.5,
    completed_steps: 500,
    target_total_steps: 500,
    dataset_total_steps: 123,
    elapsed_seconds: 250,
  })

  assert.equal(overview.layerCount, 2)
  assert.equal(overview.architectureSummary, '2x standard / swiglu')
  assert.equal(overview.formattedElapsed, '04:10')
  assert.equal(overview.formattedEta, '--')
})

test('hydrateExperiment backfills new fields for legacy saved experiments', () => {
  const experiment = hydrateExperiment({
    id: 'legacy',
    name: 'Legacy',
    timestamp: 1000,
    colorIndex: 1,
    model_config: {
      hidden_size: 128,
      vocab_size: 3204,
      max_seq_len: 64,
      rms_norm_eps: 1e-6,
      share_embedding_head: false,
      layers: [{ attention_type: 'standard', attention_params: {}, ffn_type: 'swiglu', ffn_params: {} }],
    },
    train_config: {
      batch_size: 4,
      learning_rate: 3e-4,
      max_steps: 500,
      warmup_steps: 25,
      data_order_seed: 0,
      model_init_seed: 0,
    },
    param_count: 1000,
    loss_history: [
      { step: 1, loss: 3, lr: 0.001 },
      { step: 2, loss: 2, lr: 0.0009 },
    ],
    final_loss: 2,
  })

  assert.equal(experiment.completed_steps, 2)
  assert.equal(experiment.target_total_steps, 500)
  assert.equal(experiment.dataset_total_steps, null)
  assert.equal(experiment.elapsed_seconds, null)
})
