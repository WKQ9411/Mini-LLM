<template>
  <div class="chart-container">
    <div class="chart-toolbar">
      <label class="smoothing-control">
        <span>Smoothing</span>
        <input type="range" min="0.01" max="0.99" step="0.01" :value="smoothing" @input="onSmoothingChange($event)" />
        <span class="smoothing-value">{{ smoothing.toFixed(2) }}</span>
      </label>
      <label class="smoothing-control">
        <span>Step Interval</span>
        <input type="range" min="1" max="100" step="1" :value="renderStepInterval" @input="onRenderStepIntervalChange($event)" />
        <span class="smoothing-value">{{ renderStepInterval }}</span>
      </label>
    </div>
    <div class="chart-wrapper">
      <div class="zoom-controls">
        <button class="zoom-btn" title="Zoom In" @click="zoomIn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/><line x1="16" y1="16" x2="21" y2="21"/></svg>
        </button>
        <button class="zoom-btn" title="Zoom Out" @click="zoomOut">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><line x1="8" y1="11" x2="14" y2="11"/><line x1="16" y1="16" x2="21" y2="21"/></svg>
        </button>
        <button class="zoom-btn zoom-btn-reset" :class="{ active: isZoomed }" title="Reset Zoom" @click="resetZoom">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
        </button>
      </div>
      <div ref="chartRef" class="chart-surface"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import type { Experiment } from '../types'
import { sampleByStepInterval, shouldTrackAxisPointerValue } from './lossChartState.js'

const props = defineProps<{
  currentLoss: Array<{ step: number; loss: number; lr: number }>
  experiments: Experiment[]
  currentTrainingColorIndex?: number
}>()

const chartRef = ref<HTMLElement>()
const smoothing = ref(0.25)
const renderStepInterval = ref(10)
let chart: echarts.ECharts | null = null
let resizeObserver: ResizeObserver | null = null
let updateFrame: number | null = null
let hoveredAxisValue: number | null = null
let isPointerInsideChart = false
let pinnedCurveName: string | null = null
let suppressLegendEvent = false
let lastPointerPixel: [number, number] | null = null
let lastRenderedChartSignature: string | null = null
const isZoomed = ref(false)

// ── 缓存 & 增量状态 ──────────────────────────────────────────

// 历史曲线缓存
let cachedExpSignature = ''
let cachedExpSeries: echarts.SeriesOption[] = []
let cachedExpHiddenNames = new Set<string>()

// 当前训练曲线：只增量累积原始数据，EMA 在采样后的数据上重新计算（与历史曲线一致）
let lastProcessedLength = 0
let accumulatedRaw: [number, number][] = []

function resetCurrentTrainingState() {
  lastProcessedLength = 0
  accumulatedRaw = []
}

function isSameLossPoint(rawPoint: [number, number] | undefined, sourcePoint: { step: number; loss: number } | undefined) {
  return rawPoint != null
    && sourcePoint != null
    && rawPoint[0] === sourcePoint.step
    && rawPoint[1] === sourcePoint.loss
}

function syncCurrentTrainingRawData() {
  if (props.currentLoss.length === 0) {
    resetCurrentTrainingState()
    return
  }

  const processedHeadMatches = isSameLossPoint(accumulatedRaw[0], props.currentLoss[0])
  const processedTailMatches = lastProcessedLength === 0
    || isSameLossPoint(accumulatedRaw[lastProcessedLength - 1], props.currentLoss[lastProcessedLength - 1])
  const canAppend = lastProcessedLength <= props.currentLoss.length
    && accumulatedRaw.length === lastProcessedLength
    && (lastProcessedLength === 0 || (processedHeadMatches && processedTailMatches))

  if (!canAppend) {
    accumulatedRaw = props.currentLoss.map((pt) => [pt.step, pt.loss])
    lastProcessedLength = props.currentLoss.length
    return
  }

  const newPoints = props.currentLoss.slice(lastProcessedLength)
  if (newPoints.length > 0) {
    for (const pt of newPoints) {
      accumulatedRaw.push([pt.step, pt.loss])
    }
    lastProcessedLength = props.currentLoss.length
  }
}

// ── 事件处理 ──────────────────────────────────────────────────

function handleAxisPointerUpdate(params: any) {
  const value = params?.axesInfo?.[0]?.value
  if (shouldTrackAxisPointerValue(isPointerInsideChart, value)) {
    hoveredAxisValue = value
  }
}

function hideChartTooltip() {
  isPointerInsideChart = false
  hoveredAxisValue = null
  lastPointerPixel = null
  chart?.setOption({
    tooltip: {
      alwaysShowContent: false,
    },
  }, { notMerge: false, lazyUpdate: false })
  chart?.dispatchAction({ type: 'hideTip' })
}

function handleChartPointerOut() {
  hideChartTooltip()
}

function handleChartMouseMove(event: any) {
  const x = event?.offsetX
  const y = event?.offsetY
  if (typeof x === 'number' && typeof y === 'number') {
    isPointerInsideChart = true
    lastPointerPixel = [x, y]
  }
}

function handleDataZoom() {
  if (!chart) return
  const option = chart.getOption()
  const zooms = (option.dataZoom as any[]) ?? []
  isZoomed.value = zooms.some(z => (z.start ?? 0) > 0.5 || (z.end ?? 100) < 99.5)
}

function handleLegendSelectChanged(params: any) {
  if (suppressLegendEvent) {
    return
  }

  const name = typeof params?.name === 'string' ? params.name : null
  if (!name || name.endsWith(' (original)')) {
    return
  }

  // 点击图例仅用于置顶，不改变选中状态，避免曲线变灰/隐藏。
  const selected = params?.selected as Record<string, boolean> | undefined
  if (selected && selected[name] === false && chart) {
    suppressLegendEvent = true
    chart.dispatchAction({ type: 'legendSelect', name })
    suppressLegendEvent = false
  }

  pinnedCurveName = name
  scheduleChartUpdate()
}

const colorPalette = [
  '#4a9eff', '#f6c344', '#5ad8a6', '#ff7a90', '#a78bfa', '#ff9f43', '#7dd3fc', '#f472b6',
  '#34d399', '#f87171', '#60a5fa', '#fbbf24', '#22d3ee', '#c084fc', '#fb7185', '#a3e635',
  '#38bdf8', '#f59e0b', '#818cf8', '#2dd4bf', '#fb923c', '#c4b5fd', '#bef264', '#06b6d4',
]

// ── 数据处理工具 ──────────────────────────────────────────────

/** 指数移动平均 (EMA) 平滑，与 TensorBoard 算法一致 */
function emaSmooth(data: [number, number][], factor: number): [number, number][] {
  if (factor === 0 || data.length === 0) return data
  const result: [number, number][] = []
  let last = data[0][1]
  for (const [x, y] of data) {
    last = factor * y + (1 - factor) * last
    result.push([x, last])
  }
  return result
}

function buildExperimentSignature() {
  return props.experiments
    .map((exp) => {
      const first = exp.loss_history[0]
      const last = exp.loss_history.at(-1)
      return [
        exp.id,
        exp.name,
        exp.colorIndex ?? 0,
        exp.loss_history.length,
        first?.step ?? '',
        first?.loss ?? '',
        last?.step ?? '',
        last?.loss ?? '',
      ].join(':')
    })
    .join('|')
}

function dataPointSignature(point: unknown) {
  if (!Array.isArray(point)) {
    return String(point ?? '')
  }
  return point
    .slice(0, 3)
    .map((value) => (typeof value === 'number' && Number.isFinite(value) ? value.toPrecision(12) : String(value ?? '')))
    .join(',')
}

function buildSeriesDataSignature(series: echarts.SeriesOption[]) {
  return series
    .map((item: any) => {
      const data = Array.isArray(item.data) ? item.data : []
      return [
        item.name ?? '',
        item.type ?? '',
        item.color ?? '',
        item.z ?? '',
        data.length,
        data.map(dataPointSignature).join(';'),
      ].join(':')
    })
    .join('|')
}

// ── 参数变更 ──────────────────────────────────────────────────

function onSmoothingChange(event: Event) {
  const nextValue = Number((event.target as HTMLInputElement).value)
  smoothing.value = Math.min(0.99, Math.max(0.01, nextValue))
  scheduleChartUpdate()
}

function onRenderStepIntervalChange(event: Event) {
  const nextValue = Number((event.target as HTMLInputElement).value)
  renderStepInterval.value = Math.min(100, Math.max(1, Math.round(nextValue)))
  scheduleChartUpdate()
}

function resetZoom() {
  if (!chart) return
  chart.dispatchAction({ type: 'dataZoom', start: 0, end: 100 })
  isZoomed.value = false
}

function zoomIn() {
  if (!chart) return
  const option = chart.getOption()
  const zooms = option.dataZoom as any[] | undefined
  if (!zooms) return
  for (const z of zooms) {
    const range = (z.end ?? 100) - (z.start ?? 0)
    const center = (z.start ?? 0) + range / 2
    const newRange = Math.max(range * 0.6, 1)
    chart.dispatchAction({
      type: 'dataZoom',
      dataZoomIndex: zooms.indexOf(z),
      start: Math.max(0, center - newRange / 2),
      end: Math.min(100, center + newRange / 2),
    })
  }
}

function zoomOut() {
  if (!chart) return
  const option = chart.getOption()
  const zooms = option.dataZoom as any[] | undefined
  if (!zooms) return
  for (const z of zooms) {
    const range = (z.end ?? 100) - (z.start ?? 0)
    const center = (z.start ?? 0) + range / 2
    const newRange = Math.min(range * 1.5, 100)
    chart.dispatchAction({
      type: 'dataZoom',
      dataZoomIndex: zooms.indexOf(z),
      start: Math.max(0, center - newRange / 2),
      end: Math.min(100, center + newRange / 2),
    })
  }
}

// ── 生命周期 ──────────────────────────────────────────────────

onMounted(() => {
  if (chartRef.value) {
    chart = echarts.init(chartRef.value)
    chart.on('updateAxisPointer', handleAxisPointerUpdate)
    chart.on('legendselectchanged', handleLegendSelectChanged)
    chart.on('datazoom', handleDataZoom)
    chart.getZr().on('mousemove', handleChartMouseMove)
    chart.getZr().on('globalout', handleChartPointerOut)
    chartRef.value.addEventListener('pointerleave', hideChartTooltip)
    resizeObserver = new ResizeObserver(() => {
      chart?.resize()
    })
    resizeObserver.observe(chartRef.value)
    window.addEventListener('resize', handleWindowResize)
    scheduleChartUpdate()
  }
})
onUnmounted(() => {
  if (updateFrame !== null) {
    cancelAnimationFrame(updateFrame)
    updateFrame = null
  }
  resizeObserver?.disconnect()
  resizeObserver = null
  window.removeEventListener('resize', handleWindowResize)
  chart?.off('updateAxisPointer', handleAxisPointerUpdate)
  chart?.off('legendselectchanged', handleLegendSelectChanged)
  chart?.off('datazoom', handleDataZoom)
  chart?.getZr().off('mousemove', handleChartMouseMove)
  chart?.getZr().off('globalout', handleChartPointerOut)
  chartRef.value?.removeEventListener('pointerleave', hideChartTooltip)
  chart?.dispose()
  chart = null
})

watch(
  () => ({
    currentLossLength: props.currentLoss.length,
    currentLossLastStep: props.currentLoss.at(-1)?.step ?? 0,
    currentLossLastLoss: props.currentLoss.at(-1)?.loss ?? null,
    currentTrainingColorIndex: props.currentTrainingColorIndex ?? 0,
    experimentSignature: buildExperimentSignature(),
  }),
  scheduleChartUpdate,
)

// ── 调度 ──────────────────────────────────────────────────────

function scheduleChartUpdate() {
  if (!chart || updateFrame !== null) {
    return
  }

  updateFrame = requestAnimationFrame(() => {
    updateFrame = null
    updateChart()
  })
}

// ── 历史曲线构建（带缓存） ────────────────────────────────────

function buildExperimentSeries(
  topCurveName: string | null,
  factor: number,
  interval: number,
): { series: echarts.SeriesOption[]; hiddenNames: Set<string> } {
  const expSignature = buildExperimentSignature()
  const cacheKey = `${expSignature}:${factor}:${interval}:${topCurveName}`

  if (cacheKey === cachedExpSignature && cachedExpSeries.length > 0) {
    return { series: cachedExpSeries, hiddenNames: cachedExpHiddenNames }
  }

  const series: echarts.SeriesOption[] = []
  const hiddenNames = new Set<string>()

  for (const exp of props.experiments) {
    const displayName = exp.name
    const color = colorPalette[(exp.colorIndex ?? 0) % colorPalette.length]
    const rawData: [number, number][] = exp.loss_history.map(d => [d.step, d.loss])
    const renderData = sampleByStepInterval(rawData, interval)
    const isTopCurve = topCurveName != null && displayName === topCurveName
    const mainZ = isTopCurve ? 20 : 10
    const rawZ = isTopCurve ? 19 : 9

    if (factor > 0) {
      const smoothData = emaSmooth(renderData, factor)
      const rawName = `${displayName} (original)`
      hiddenNames.add(rawName)
      series.push({
        name: rawName,
        type: 'line',
        data: renderData,
        smooth: false,
        color,
        lineStyle: { width: 1, opacity: 0.12, color },
        z: rawZ,
        symbol: 'none',
        animation: false,
        tooltip: { show: false },
      })
      series.push({
        name: displayName,
        type: 'line',
        data: smoothData.map((p, i) => [p[0], p[1], renderData[i][1]]),
        smooth: false,
        color,
        lineStyle: { width: 1.8, color },
        z: mainZ,
        symbol: 'none',
        animation: false,
      })
    } else {
      series.push({
        name: displayName,
        type: 'line',
        data: renderData,
        smooth: false,
        color,
        lineStyle: { width: 1.8, color },
        z: mainZ,
        symbol: 'none',
        animation: false,
      })
    }
  }

  cachedExpSignature = cacheKey
  cachedExpSeries = series
  cachedExpHiddenNames = hiddenNames
  return { series, hiddenNames }
}

// ── 当前训练曲线构建（增量累积原始数据，采样+EMA 与历史曲线一致） ──

function buildCurrentSeries(
  factor: number,
  interval: number,
  isTopCurve: boolean,
): { series: echarts.SeriesOption[]; hiddenNames: Set<string> } {
  const series: echarts.SeriesOption[] = []
  const hiddenNames = new Set<string>()
  const displayName = 'Current Training'
  const color = colorPalette[(props.currentTrainingColorIndex ?? 0) % colorPalette.length]
  const mainZ = isTopCurve ? 20 : 10
  const rawZ = isTopCurve ? 19 : 9

  // 正常训练只增量追加；新训练或数据替换时重建，避免旧点残留。
  syncCurrentTrainingRawData()

  // 当前训练曲线不保留非 interval 尾点；31-39 停留在 30，40 时再更新。
  const renderData = sampleByStepInterval(accumulatedRaw, interval, false)
  const currentSymbol = renderData.length === 1 ? 'circle' : 'none'

  if (factor > 0) {
    const smoothData = emaSmooth(renderData, factor)

    const rawName = `${displayName} (original)`
    hiddenNames.add(rawName)
    series.push({
      name: rawName,
      type: 'line',
      data: renderData,
      smooth: false,
      color,
      lineStyle: { width: 1, opacity: 0.12, color },
      z: rawZ,
      symbol: currentSymbol,
      symbolSize: 6,
      animation: false,
      tooltip: { show: false },
    })
    series.push({
      name: displayName,
      type: 'line',
      data: smoothData.map((p, i) => [p[0], p[1], renderData[i][1]]),
      smooth: false,
      color,
      lineStyle: { width: 1.8, color },
      z: mainZ,
      symbol: currentSymbol,
      symbolSize: 6,
      animation: false,
    })

    if (smoothData.length > 0) {
      const latest = smoothData[smoothData.length - 1]
      const latestName = `${displayName} (latest)`
      hiddenNames.add(latestName)
      series.push({
        name: latestName,
        type: 'effectScatter',
        data: [[latest[0], latest[1], renderData[renderData.length - 1][1]]],
        symbolSize: 7,
        showEffectOn: 'render',
        rippleEffect: {
          brushType: 'stroke',
          scale: 2.8,
          period: 2.2,
        },
        itemStyle: { color },
        z: 40,
        tooltip: { show: false },
        animation: false,
      })
    }
  } else {
    series.push({
      name: displayName,
      type: 'line',
      data: renderData,
      smooth: false,
      color,
      lineStyle: { width: 1.8, color },
      z: mainZ,
      symbol: currentSymbol,
      symbolSize: 6,
      animation: false,
    })

    if (renderData.length > 0) {
      const latest = renderData[renderData.length - 1]
      const latestName = `${displayName} (latest)`
      hiddenNames.add(latestName)
      series.push({
        name: latestName,
        type: 'effectScatter',
        data: [[latest[0], latest[1], latest[1]]],
        symbolSize: 7,
        showEffectOn: 'render',
        rippleEffect: {
          brushType: 'stroke',
          scale: 2.8,
          period: 2.2,
        },
        itemStyle: { color },
        z: 40,
        tooltip: { show: false },
        animation: false,
      })
    }
  }

  return { series, hiddenNames }
}

// ── 主渲染 ────────────────────────────────────────────────────

function updateChart() {
  if (!chart) return

  const factor = smoothing.value
  const interval = renderStepInterval.value
  const hasCurrentTrainingCurve = props.currentLoss.length > 0

  if (pinnedCurveName && !hasCurrentTrainingCurve && !props.experiments.some(exp => exp.name === pinnedCurveName)) {
    pinnedCurveName = null
  }

  const topCurveName = hasCurrentTrainingCurve ? 'Current Training' : pinnedCurveName

  // 检测 currentLoss 被清空或截断（新训练开始/训练结束）→ 重置增量状态
  if (!hasCurrentTrainingCurve || props.currentLoss.length < lastProcessedLength) {
    resetCurrentTrainingState()
  }

  const allSeries: echarts.SeriesOption[] = []
  const allHiddenNames = new Set<string>()
  let currentSeriesSignature = ''

  // 1) 历史曲线（带缓存）
  const expResult = buildExperimentSeries(topCurveName, factor, interval)
  allSeries.push(...expResult.series)
  for (const n of expResult.hiddenNames) allHiddenNames.add(n)

  // 2) 当前训练曲线（增量）
  if (hasCurrentTrainingCurve) {
    const isTopCurve = topCurveName === 'Current Training'
    const curResult = buildCurrentSeries(factor, interval, isTopCurve)
    allSeries.push(...curResult.series)
    for (const n of curResult.hiddenNames) allHiddenNames.add(n)
    currentSeriesSignature = buildSeriesDataSignature(curResult.series)
  }

  const chartSignature = [
    buildExperimentSignature(),
    factor,
    interval,
    topCurveName ?? '',
    props.currentTrainingColorIndex ?? 0,
    allSeries.length,
    Array.from(allHiddenNames).sort().join(','),
    currentSeriesSignature,
  ].join('|')

  if (chartSignature === lastRenderedChartSignature) {
    return
  }
  lastRenderedChartSignature = chartSignature

  if (allSeries.length === 0) {
    chart.clear()
  }

  chart.setOption({
    backgroundColor: 'transparent',
    animation: false,
    tooltip: {
      trigger: 'axis',
      alwaysShowContent: isPointerInsideChart,
      transitionDuration: 0,
      axisPointer: { animation: false },
      formatter(params: any) {
        if (!Array.isArray(params) || params.length === 0) return ''
        const axisValue = Number(params[0].axisValue)
        const step = Number.isFinite(axisValue) ? Math.round(axisValue) : Math.round(params[0].data?.[0] ?? 0)
        let html = `<div style="font-size:11px;margin-bottom:4px;color:#999">Step ${step}</div>`
        for (const p of params) {
          const smoothValue = typeof p.data?.[1] === 'number' ? p.data[1] : null
          const rawValue = typeof p.data?.[2] === 'number' ? p.data[2] : null
          if (smoothValue == null && rawValue == null) {
            continue
          }

          const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${p.color};margin-right:6px;vertical-align:middle"></span>`
          const smoothVal = smoothValue != null ? smoothValue.toFixed(4) : '--'
          const rawVal = rawValue != null ? rawValue.toFixed(4) : null
          if (rawVal !== null) {
            html += `<div style="font-size:11px;line-height:18px">${dot}${p.seriesName}&ensp;smooth: <b>${smoothVal}</b>&ensp;original: <b>${rawVal}</b></div>`
          } else {
            html += `<div style="font-size:11px;line-height:18px">${dot}${p.seriesName}&ensp;<b>${smoothVal}</b></div>`
          }
        }
        return html
      }
    },
    legend: {
      top: 0,
      textStyle: { color: '#aaa', fontSize: 11 },
      data: allSeries
        .map((s: any) => s.name as string)
        .filter((name: string) => !allHiddenNames.has(name)),
    },
    grid: { left: 60, right: 90, top: 40, bottom: 70 },
    xAxis: { type: 'value', name: 'Step', axisLine: { lineStyle: { color: '#555' } }, splitLine: { lineStyle: { color: '#222' } } },
    yAxis: { type: 'value', name: 'Loss', axisLine: { lineStyle: { color: '#555' } }, splitLine: { lineStyle: { color: '#222' } } },
    dataZoom: [
      { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
      { type: 'inside', yAxisIndex: 0, filterMode: 'none' },
      { type: 'slider', xAxisIndex: 0, height: 20, bottom: 10, borderColor: '#444', fillerColor: 'rgba(74,158,255,0.15)', handleStyle: { color: '#4a9eff' }, textStyle: { color: '#888' }, filterMode: 'none' },
      { type: 'slider', yAxisIndex: 0, width: 20, right: 10, borderColor: '#444', fillerColor: 'rgba(74,158,255,0.15)', handleStyle: { color: '#4a9eff' }, textStyle: { color: '#888' }, filterMode: 'none' },
    ],
    series: allSeries,
  }, { notMerge: false, replaceMerge: ['series'], lazyUpdate: true })

  if (isPointerInsideChart && hoveredAxisValue != null) {
    if (lastPointerPixel) {
      chart.dispatchAction({
        type: 'showTip',
        x: lastPointerPixel[0],
        y: lastPointerPixel[1],
      })
    } else {
      chart.dispatchAction({
        type: 'showTip',
        xAxisIndex: 0,
        value: hoveredAxisValue,
      })
    }
  }
}

function handleWindowResize() {
  chart?.resize()
}
</script>
