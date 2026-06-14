import test from 'node:test'
import assert from 'node:assert/strict'

import {
  connectTrainWebSocket,
  deleteExperiment,
  estimateParams,
  fetchExperiments,
  fetchGpuInfo,
  saveExperiment,
  stopTraining,
  startTraining,
} from './index.ts'

test('estimateParams rejects non-ok responses', async () => {
  global.fetch = async () => ({
    ok: false,
    json: async () => ({ error: 'bad config' }),
  })

  await assert.rejects(
    () => estimateParams({ hidden_size: 1, vocab_size: 1, max_seq_len: 1, rms_norm_eps: 1e-6, share_embedding_head: false, layers: [] }),
    /bad config/,
  )
})

test('startTraining rejects non-ok responses', async () => {
  global.fetch = async () => ({
    ok: false,
    json: async () => ({ error: 'already training' }),
  })

  await assert.rejects(
    () => startTraining({ hidden_size: 1, vocab_size: 1, max_seq_len: 1, rms_norm_eps: 1e-6, share_embedding_head: false, layers: [] }, { batch_size: 1, learning_rate: 1e-3, max_steps: 1, warmup_steps: 0, data_order_seed: 0, model_init_seed: 0 }),
    /already training/,
  )
})

test('fetchExperiments returns persisted experiments', async () => {
  const payload = [{ id: '1', name: 'Exp 1' }]
  global.fetch = async () => ({
    ok: true,
    text: async () => JSON.stringify(payload),
  })

  await assert.deepEqual(await fetchExperiments(), payload)
})

test('fetchGpuInfo returns GPU monitor payload', async () => {
  const payload = {
    available: true,
    source: 'nvidia-smi',
    message: null,
    gpus: [{ index: 0, name: 'GPU', memory_total_mb: 1000, memory_used_mb: 250, utilization_pct: 50, temperature_c: 60 }],
  }
  global.fetch = async (url) => {
    assert.equal(url, '/api/gpu')
    return {
      ok: true,
      text: async () => JSON.stringify(payload),
    }
  }

  await assert.deepEqual(await fetchGpuInfo(), payload)
})

test('saveExperiment posts experiment payload', async () => {
  const requests = []
  const payload = { id: '1', name: 'Exp 1' }
  global.fetch = async (url, init) => {
    requests.push({ url, init })
    return {
      ok: true,
      text: async () => JSON.stringify(payload),
    }
  }

  const saved = await saveExperiment(payload)

  assert.deepEqual(saved, payload)
  assert.equal(requests[0].url, '/api/experiments')
  assert.equal(requests[0].init.method, 'POST')
  assert.deepEqual(JSON.parse(requests[0].init.body), { data: payload })
})

test('deleteExperiment issues delete request', async () => {
  const requests = []
  global.fetch = async (url, init) => {
    requests.push({ url, init })
    return {
      ok: true,
      text: async () => '',
    }
  }

  await deleteExperiment('exp-1')

  assert.equal(requests[0].url, '/api/experiments/exp-1')
  assert.equal(requests[0].init.method, 'DELETE')
})

test('stopTraining issues stop request', async () => {
  const requests = []
  global.fetch = async (url, init) => {
    requests.push({ url, init })
    return {
      ok: true,
      text: async () => '',
    }
  }

  await stopTraining()

  assert.equal(requests[0].url, '/api/stop')
  assert.equal(requests[0].init.method, 'POST')
})

test('connectTrainWebSocket waits until the socket is open', async () => {
  const received = []

  class FakeWebSocket {
    constructor(url) {
      this.url = url
      this.onopen = null
      this.onmessage = null
      queueMicrotask(() => {
        this.readyState = 1
        this.onopen?.()
        this.onmessage?.({ data: JSON.stringify({ type: 'step', step: 1, loss: 1.5, lr: 0.1 }) })
      })
    }
  }

  global.location = { protocol: 'http:', host: 'localhost:5173' }
  global.WebSocket = FakeWebSocket

  const pending = connectTrainWebSocket((msg) => received.push(msg))
  assert.equal(typeof pending?.then, 'function')

  const ws = await pending

  assert.equal(ws.url, 'ws://localhost:5173/ws/train')
  assert.deepEqual(received, [{ type: 'step', step: 1, loss: 1.5, lr: 0.1 }])
})
