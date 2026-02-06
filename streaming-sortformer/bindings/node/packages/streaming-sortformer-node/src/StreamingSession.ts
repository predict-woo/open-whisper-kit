/**
 * TypeScript wrapper for native StreamingSession
 */

import type { FeedResult, StreamingPreset } from './types.js';

/**
 * Streaming diarization session
 *
 * Maintains state across incremental audio feed calls for true real-time
 * speaker diarization. State is kept in native C code for efficiency.
 *
 * @example
 * ```typescript
 * const session = model.createStreamingSession({ preset: '2s' });
 *
 * // Feed audio chunks as they arrive
 * const result1 = session.feed(chunk1);
 * const result2 = session.feed(chunk2);
 *
 * // Get total frames processed
 * console.log(session.totalFrames);
 *
 * // Reset for new audio stream
 * session.reset();
 *
 * // Clean up
 * session.close();
 * ```
 */
export class StreamingSession {
  private native: any;
  private _closed: boolean = false;

  /**
   * Create a new streaming session
   * @param native - Native StreamingSession instance from binding
   * @internal
   */
  constructor(native: any) {
    this.native = native;
  }

  /**
   * Feed audio samples and get predictions for this chunk
   *
   * @param audio - Audio samples as Float32Array (16kHz mono)
   * @returns Predictions for the new frames in this chunk
   * @throws Error if session is closed or audio is invalid
   *
   * @example
   * ```typescript
   * const audio = new Float32Array(48000); // 3 seconds
   * const result = session.feed(audio);
   * console.log(`Got ${result.frameCount} new frames`);
   * ```
   */
  feed(audio: Float32Array): FeedResult {
    if (this._closed) {
      throw new Error('Session is closed');
    }

    if (!(audio instanceof Float32Array)) {
      throw new TypeError('audio must be a Float32Array');
    }

    const result = this.native.feed(audio);

    return {
      predictions: result.predictions,
      frameCount: result.frameCount,
    };
  }

  /**
   * Flush remaining buffered audio at end of stream
   *
   * Call this when the audio stream ends to process any remaining
   * buffered audio that hasn't been output yet due to latency buffering.
   *
   * @returns Final predictions for buffered audio
   * @throws Error if session is closed
   */
  flush(): FeedResult {
    if (this._closed) {
      throw new Error('Session is closed');
    }

    const result = this.native.flush();

    return {
      predictions: result.predictions,
      frameCount: result.frameCount,
    };
  }

  /**
   * Reset the streaming state for a new audio stream
   *
   * Clears all internal buffers (spkcache, fifo, mel overlap) while
   * keeping the model loaded. Use this when starting a new recording.
   *
   * @throws Error if session is closed
   */
  reset(): void {
    if (this._closed) {
      throw new Error('Session is closed');
    }
    this.native.reset();
  }

  /**
   * Close the session and free native resources
   *
   * After calling close(), the session cannot be used.
   * Calling close() multiple times is safe (idempotent).
   */
  close(): void {
    if (!this._closed) {
      if (this.native && typeof this.native.close === 'function') {
        this.native.close();
      }
      this._closed = true;
    }
  }

  /**
   * Get total frames output so far
   */
  get totalFrames(): number {
    if (this._closed) {
      return 0;
    }
    return this.native.getTotalFrames();
  }

  /**
   * Check if the session is closed
   */
  get isClosed(): boolean {
    return this._closed;
  }
}
