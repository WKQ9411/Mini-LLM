import test from 'node:test'
import assert from 'node:assert/strict'

import { sampleByStepInterval, shouldTrackAxisPointerValue } from './lossChartState.js'

test('current training sampling includes interval-aligned latest step without trailing non-interval points', () => {
  const dataAt39 = Array.from({ length: 39 }, (_, index) => [index + 1, 10 - index])
  const dataAt40 = Array.from({ length: 40 }, (_, index) => [index + 1, 10 - index])

  assert.deepEqual(
    sampleByStepInterval(dataAt39, 10, false).map(([step]) => step),
    [1, 10, 20, 30],
  )
  assert.deepEqual(
    sampleByStepInterval(dataAt40, 10, false).map(([step]) => step),
    [1, 10, 20, 30, 40],
  )
})

test('history sampling keeps first interval points and final trailing step', () => {
  const data = Array.from({ length: 39 }, (_, index) => [index + 1, 10 - index])

  assert.deepEqual(
    sampleByStepInterval(data, 10, true).map(([step]) => step),
    [1, 10, 20, 30, 39],
  )
})

test('axis pointer updates do not reactivate tooltip after pointer leaves chart', () => {
  assert.equal(shouldTrackAxisPointerValue(false, 30), false)
  assert.equal(shouldTrackAxisPointerValue(true, 30), true)
  assert.equal(shouldTrackAxisPointerValue(true, Number.NaN), false)
})
