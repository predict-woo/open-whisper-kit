/**
 * Latency presets for streaming diarization
 * Maps preset names to their corresponding parameter configurations
 */

import type { LatencyPreset } from './types';

/**
 * Streaming latency preset parameters
 * Each preset controls chunk processing, buffering, and speaker cache update behavior
 */
export interface PresetParams {
  /** Chunk length in frames (16kHz, hop=160) */
  chunkLen: number;
  /** Right context frames for conformer processing */
  rightContext: number;
  /** FIFO buffer length in frames */
  fifoLen: number;
  /** Speaker cache update period in frames */
  spkcacheUpdatePeriod: number;
}

/**
 * Streaming latency presets
 * - 'low': ~188ms latency, minimal buffering
 * - '2s': ~2 second latency
 * - '3s': ~3 second latency
 * - '5s': ~5 second latency
 */
export const LATENCY_PRESETS: Record<LatencyPreset, PresetParams> = {
  'low': {
    chunkLen: 6,
    rightContext: 7,
    fifoLen: 188,
    spkcacheUpdatePeriod: 144,
  },
  '2s': {
    chunkLen: 15,
    rightContext: 10,
    fifoLen: 100,
    spkcacheUpdatePeriod: 144,
  },
  '3s': {
    chunkLen: 30,
    rightContext: 7,
    fifoLen: 100,
    spkcacheUpdatePeriod: 100,
  },
  '5s': {
    chunkLen: 55,
    rightContext: 7,
    fifoLen: 100,
    spkcacheUpdatePeriod: 100,
  },
} as const;

/**
 * Offline mode parameters
 * Used when mode='offline' to process entire audio at once
 */
export const OFFLINE_PARAMS: PresetParams = {
  chunkLen: 188,
  rightContext: 1,
  fifoLen: 0,
  spkcacheUpdatePeriod: 188,
} as const;

/**
 * Get preset parameters by name
 * @param preset - Latency preset name
 * @returns Preset parameters
 * @throws Error if preset is not found
 */
export function getPresetParams(preset: LatencyPreset): PresetParams {
  const params = LATENCY_PRESETS[preset];
  if (!params) {
    throw new Error(`Unknown latency preset: ${preset}`);
  }
  return params;
}

/**
 * Get default preset parameters for offline mode
 * @returns Offline mode parameters
 */
export function getOfflineParams(): PresetParams {
  return OFFLINE_PARAMS;
}
