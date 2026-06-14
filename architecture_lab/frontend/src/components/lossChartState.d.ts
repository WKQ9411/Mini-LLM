export function sampleByStepInterval(
  data: Array<[number, number]>,
  interval: number,
  includeTrailingPoint?: boolean,
): Array<[number, number]>

export function shouldTrackAxisPointerValue(
  pointerInsideChart: boolean,
  value: unknown,
): boolean
