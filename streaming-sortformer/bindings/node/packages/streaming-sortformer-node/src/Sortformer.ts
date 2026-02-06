/**
 * TypeScript wrapper for the native SortFormer speaker diarization model
 */

import type { LoadOptions, DiarizeOptions, DiarizeResult, StreamingSessionOptions, StreamingPreset } from './types.js';
import { LATENCY_PRESETS, OFFLINE_PARAMS } from './presets.js';
import { getBinding } from './binding.js';
import { StreamingSession } from './StreamingSession.js';

/**
 * SortFormer speaker diarization model wrapper
 *
 * Provides a high-level TypeScript API for loading and running the native
 * SortFormer model for streaming speaker diarization.
 *
 * @example
 * ```typescript
 * const model = await Sortformer.load('./model.gguf', { threads: 4 });
 * const result = await model.diarize(audioData, { mode: 'streaming', latency: '2s' });
 * console.log(result.rttm);
 * model.close();
 * ```
 */
export class Sortformer {
  private native: any;
  private closed: boolean = false;

  /**
   * Private constructor - use static load() method instead
   * @param native - Native SortformerModel instance from binding
   */
  private constructor(native: any) {
    this.native = native;
  }

  /**
   * Load a SortFormer model from a GGUF file
   *
   * @param modelPath - Path to the GGUF model file
   * @param options - Optional loading configuration
   * @returns Promise resolving to a loaded Sortformer instance
   * @throws Error if model file not found or native binding unavailable
   *
   * @example
   * ```typescript
   * const model = await Sortformer.load('./model.gguf', { threads: 8 });
   * ```
   */
  static async load(modelPath: string, options?: LoadOptions): Promise<Sortformer> {
    // Validate input
    if (!modelPath || typeof modelPath !== 'string') {
      throw new TypeError('modelPath must be a non-empty string');
    }

    // Get native binding
    const binding = getBinding();

    // Create native model instance
    // Default to 4 threads if not specified
    const threads = options?.threads ?? 4;

    if (threads < 1 || !Number.isInteger(threads)) {
      throw new Error('threads must be a positive integer');
    }

    // Instantiate native model
    const native = new binding.SortformerModel(modelPath, threads);

    return new Sortformer(native);
  }

  /**
   * Run diarization inference on audio samples
   *
   * @param audio - Audio samples as Float32Array (16kHz mono)
   * @param options - Optional diarization configuration
   * @returns Promise resolving to diarization results (RTTM + predictions)
   * @throws Error if model is closed, audio is invalid, or inference fails
   *
   * @example
   * ```typescript
   * const result = await model.diarize(audioData, {
   *   mode: 'streaming',
   *   latency: '2s',
   *   threshold: 0.5,
   *   medianFilter: 11
   * });
   * ```
   */
  async diarize(audio: Float32Array, options?: DiarizeOptions): Promise<DiarizeResult> {
    // Check if model is closed
    if (this.closed) {
      throw new Error('Model is closed. Cannot perform diarization.');
    }

    // Validate audio input
    if (!(audio instanceof Float32Array)) {
      throw new TypeError('audio must be a Float32Array');
    }

    if (audio.length === 0) {
      throw new Error('audio cannot be empty');
    }

    // Validate options
    if (options?.threshold !== undefined) {
      if (typeof options.threshold !== 'number' || options.threshold < 0 || options.threshold > 1) {
        throw new Error('threshold must be a number between 0 and 1');
      }
    }

    if (options?.medianFilter !== undefined) {
      if (!Number.isInteger(options.medianFilter) || options.medianFilter < 1 || options.medianFilter % 2 === 0) {
        throw new Error('medianFilter must be a positive odd integer');
      }
    }

    // Map user-friendly options to native format
    const mode = options?.mode ?? 'offline';
    const nativeOptions: any = {
      threshold: options?.threshold ?? 0.5,
      medianFilter: options?.medianFilter ?? 11,
    };

    // Add streaming-specific parameters if in streaming mode
    if (mode === 'streaming') {
      const latency = options?.latency ?? '2s';
      const presetParams = LATENCY_PRESETS[latency];

      if (!presetParams) {
        throw new Error(`Unknown latency preset: ${latency}`);
      }

      nativeOptions.chunkLen = presetParams.chunkLen;
      nativeOptions.rightContext = presetParams.rightContext;
      nativeOptions.fifoLen = presetParams.fifoLen;
      nativeOptions.spkcacheUpdatePeriod = presetParams.spkcacheUpdatePeriod;
    } else if (mode === 'offline') {
      // Use offline parameters
      nativeOptions.chunkLen = OFFLINE_PARAMS.chunkLen;
      nativeOptions.rightContext = OFFLINE_PARAMS.rightContext;
      nativeOptions.fifoLen = OFFLINE_PARAMS.fifoLen;
      nativeOptions.spkcacheUpdatePeriod = OFFLINE_PARAMS.spkcacheUpdatePeriod;
    } else {
      throw new Error(`Unknown diarization mode: ${mode}`);
    }

    // Call native diarization
    const result = await this.native.diarize(audio, nativeOptions);

    // Validate result structure
    if (!result || typeof result !== 'object') {
      throw new Error('Native diarization returned invalid result');
    }

    if (typeof result.rttm !== 'string') {
      throw new Error('Native diarization result missing rttm string');
    }

    if (!(result.predictions instanceof Float32Array)) {
      throw new Error('Native diarization result predictions must be Float32Array');
    }

    if (!Number.isInteger(result.frameCount) || result.frameCount < 0) {
      throw new Error('Native diarization result frameCount must be non-negative integer');
    }

    if (!Number.isInteger(result.speakerCount) || result.speakerCount < 1 || result.speakerCount > 4) {
      throw new Error('Native diarization result speakerCount must be 1-4');
    }

    return result as DiarizeResult;
  }

  /**
   * Close the model and free native resources
   *
   * After calling close(), the model cannot be used for further inference.
   * Calling close() multiple times is safe (idempotent).
   *
   * @example
   * ```typescript
   * model.close();
   * ```
   */
  close(): void {
    if (!this.closed) {
      if (this.native && typeof this.native.close === 'function') {
        this.native.close();
      }
      this.closed = true;
    }
  }

  /**
   * Check if the model is closed
   * @returns true if the model has been closed, false otherwise
   */
  isClosed(): boolean {
    return this.closed;
  }

  /**
   * Create a streaming session for incremental audio processing
   *
   * The streaming session maintains state (speaker cache, FIFO buffer)
   * across feed() calls, enabling true real-time diarization.
   *
   * @param options - Optional streaming configuration
   * @returns A new StreamingSession instance
   * @throws Error if model is closed
   *
   * @example
   * ```typescript
   * const session = model.createStreamingSession({ preset: 'low' });
   *
   * // Feed audio chunks as they arrive
   * const result1 = session.feed(chunk1);
   * const result2 = session.feed(chunk2);
   *
   * // Accumulate predictions
   * const allPreds = [...result1.predictions, ...result2.predictions];
   *
   * session.close();
   * ```
   */
  createStreamingSession(options?: StreamingSessionOptions): StreamingSession {
    if (this.closed) {
      throw new Error('Model is closed. Cannot create streaming session.');
    }

    const preset = options?.preset ?? '2s';

    // Map preset string to enum value
    const presetMap: Record<StreamingPreset, number> = {
      'low': 0,  // SORTFORMER_PRESET_LOW_LATENCY
      '2s': 1,   // SORTFORMER_PRESET_2S
      '3s': 2,   // SORTFORMER_PRESET_3S
      '5s': 3,   // SORTFORMER_PRESET_5S
    };

    const presetNum = presetMap[preset];
    if (presetNum === undefined) {
      throw new Error(`Unknown preset: ${preset}`);
    }

    // Get binding and create native session
    const binding = getBinding();
    const nativeSession = new binding.StreamingSession(this.native, presetNum);

    return new StreamingSession(nativeSession);
  }
}
