/**
 * TypeScript type definitions for streaming-sortformer-node
 */

/**
 * Diarization mode: offline processes entire audio at once,
 * streaming processes audio in chunks with latency control
 */
export type DiarizeMode = 'offline' | 'streaming';

/**
 * Latency preset for streaming mode
 * - 'low': ~188ms latency, minimal buffering
 * - '2s': ~2 second latency
 * - '3s': ~3 second latency
 * - '5s': ~5 second latency
 */
export type LatencyPreset = 'low' | '2s' | '3s' | '5s';

/**
 * Options for loading a SortFormer model
 */
export interface LoadOptions {
  /**
   * Number of CPU threads to use for inference
   * @default auto-detected based on CPU cores
   */
  threads?: number;
}

/**
 * Options for diarization inference
 */
export interface DiarizeOptions {
  /**
   * Diarization mode: 'offline' or 'streaming'
   * @default 'offline'
   */
  mode?: DiarizeMode;

  /**
   * Latency preset for streaming mode
   * Only used when mode='streaming'
   * @default '2s'
   */
  latency?: LatencyPreset;

  /**
   * Speaker activity threshold (0.0 to 1.0)
   * Frames with prediction >= threshold are considered active
   * @default 0.5
   */
  threshold?: number;

  /**
   * Median filter window size for smoothing predictions
   * Must be odd number >= 1
   * @default 11
   */
  medianFilter?: number;
}

/**
 * Result of diarization inference
 */
export interface DiarizeResult {
  /**
   * RTTM format output (speaker diarization segments)
   * Format: SPEAKER <filename> <channel> <start> <duration> <conf> <spk_type> <spk_id> <score>
   */
  rttm: string;

  /**
   * Raw per-frame speaker activity predictions
   * Shape: [frameCount, 4] (4 speakers max)
   * Values: 0.0 to 1.0 (probability of speaker activity)
   */
  predictions: Float32Array;

  /**
   * Number of frames in the output
   */
  frameCount: number;

  /**
   * Number of speakers detected (1-4)
   */
  speakerCount: number;
}

/**
 * Streaming preset type
 */
export type StreamingPreset = 'low' | '2s' | '3s' | '5s';

/**
 * Options for creating a streaming session
 */
export interface StreamingSessionOptions {
  /**
   * Latency preset
   * @default '2s'
   */
  preset?: StreamingPreset;
}

/**
 * Result from feeding audio to streaming session
 */
export interface FeedResult {
  /**
   * Per-frame speaker predictions for this chunk
   * Shape: [frameCount, 4]
   */
  predictions: Float32Array;

  /**
   * Number of new frames in this result
   */
  frameCount: number;
}
