export function sampleByStepInterval(data, interval, includeTrailingPoint = true) {
  if (data.length <= 1 || interval <= 1) {
    return data
  }

  const sampled = []
  const first = data[0]
  sampled.push(first)

  for (let i = 1; i < data.length; i++) {
    const point = data[i]
    const isLast = i === data.length - 1
    const isIntervalPoint = point[0] % interval === 0
    const isTrailingPoint = isLast && includeTrailingPoint && point[0] !== first[0]

    if (isIntervalPoint || isTrailingPoint) {
      sampled.push(point)
    }
  }

  return sampled
}

export function shouldTrackAxisPointerValue(pointerInsideChart, value) {
  return pointerInsideChart && typeof value === 'number' && Number.isFinite(value)
}
