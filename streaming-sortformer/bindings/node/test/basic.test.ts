import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Sortformer } from '../packages/streaming-sortformer-node/src';
import type { LatencyPreset } from '../packages/streaming-sortformer-node/src';
import * as fs from 'fs';
import * as path from 'path';

// Paths relative to repo root
const REPO_ROOT = path.resolve(__dirname, '../../..');
const MODEL_PATH = path.join(REPO_ROOT, 'model.gguf');
const TEST_WAV_PATH = path.join(REPO_ROOT, 'test.wav');

// Check if model exists for conditional test skipping
const modelExists = fs.existsSync(MODEL_PATH);
const testWavExists = fs.existsSync(TEST_WAV_PATH);

// Check if native binding is available
let bindingAvailable = false;
try {
  // Try to load the binding - this will throw if not available
  const { getBinding } = await import('../packages/streaming-sortformer-node/src/binding');
  getBinding();
  bindingAvailable = true;
} catch {
  bindingAvailable = false;
}

// Combined check: both model and binding must be available
const canRunNativeTests = modelExists && bindingAvailable;

/**
 * Read WAV file and return Float32Array of samples
 * Assumes 16kHz mono 16-bit PCM WAV
 */
function readWavFile(filePath: string): Float32Array {
  const buffer = fs.readFileSync(filePath);
  
  // Parse WAV header (44 bytes for standard PCM WAV)
  // Skip to data chunk
  let dataOffset = 44;
  
  // Find 'data' chunk
  for (let i = 12; i < buffer.length - 8; i++) {
    if (buffer.toString('ascii', i, i + 4) === 'data') {
      dataOffset = i + 8; // Skip 'data' + size (4 bytes each)
      break;
    }
  }
  
  // Read 16-bit samples and convert to float32
  const numSamples = Math.floor((buffer.length - dataOffset) / 2);
  const samples = new Float32Array(numSamples);
  
  for (let i = 0; i < numSamples; i++) {
    const sample = buffer.readInt16LE(dataOffset + i * 2);
    samples[i] = sample / 32768.0; // Normalize to [-1, 1]
  }
  
  return samples;
}

describe('Sortformer', () => {
  // Log skip reasons
  beforeAll(() => {
    if (!modelExists) {
      console.warn(`Skipping native tests: model.gguf not found at ${MODEL_PATH}`);
    }
    if (!bindingAvailable) {
      console.warn('Skipping native tests: native binding not available (not built yet)');
    }
  });

  describe('Model Loading', () => {
    it.skipIf(!canRunNativeTests)('loads model successfully', async () => {
      const model = await Sortformer.load(MODEL_PATH);
      expect(model).toBeDefined();
      expect(model.isClosed()).toBe(false);
      model.close();
    });

    it.skipIf(!canRunNativeTests)('loads model with custom thread count', async () => {
      const model = await Sortformer.load(MODEL_PATH, { threads: 2 });
      expect(model).toBeDefined();
      model.close();
    });

    it('throws error for invalid model path', async () => {
      // This test should throw either because binding is not available
      // or because the model path is invalid
      await expect(Sortformer.load('/nonexistent/path/model.gguf')).rejects.toThrow();
    });

    it('throws error for empty model path', async () => {
      await expect(Sortformer.load('')).rejects.toThrow('modelPath must be a non-empty string');
    });

    it.skipIf(!canRunNativeTests)('throws error for invalid thread count', async () => {
      await expect(Sortformer.load(MODEL_PATH, { threads: 0 })).rejects.toThrow('threads must be a positive integer');
      await expect(Sortformer.load(MODEL_PATH, { threads: -1 })).rejects.toThrow('threads must be a positive integer');
      await expect(Sortformer.load(MODEL_PATH, { threads: 1.5 })).rejects.toThrow('threads must be a positive integer');
    });
  });

  describe('Diarization', () => {
    let model: Sortformer;

    beforeAll(async () => {
      if (canRunNativeTests) {
        model = await Sortformer.load(MODEL_PATH, { threads: 4 });
      }
    });

    afterAll(() => {
      model?.close();
    });

    it.skipIf(!canRunNativeTests)('diarizes synthetic audio and returns RTTM string', async () => {
      // 5 seconds of silence at 16kHz
      const audio = new Float32Array(16000 * 5);
      const result = await model.diarize(audio);
      
      expect(result).toBeDefined();
      expect(typeof result.rttm).toBe('string');
    });

    it.skipIf(!canRunNativeTests)('diarizes synthetic audio and returns Float32Array predictions', async () => {
      const audio = new Float32Array(16000 * 5);
      const result = await model.diarize(audio);
      
      expect(result.predictions).toBeInstanceOf(Float32Array);
      expect(result.frameCount).toBeGreaterThan(0);
      expect(result.speakerCount).toBeGreaterThanOrEqual(1);
      expect(result.speakerCount).toBeLessThanOrEqual(4);
    });

    it.skipIf(!canRunNativeTests || !testWavExists)('diarizes real audio file', async () => {
      const audio = readWavFile(TEST_WAV_PATH);
      const result = await model.diarize(audio, { mode: 'streaming', latency: '2s' });
      
      expect(result.rttm).toBeDefined();
      expect(result.predictions).toBeInstanceOf(Float32Array);
      expect(result.frameCount).toBeGreaterThan(0);
    });

    it.skipIf(!canRunNativeTests)('throws error for empty audio', async () => {
      const emptyAudio = new Float32Array(0);
      await expect(model.diarize(emptyAudio)).rejects.toThrow('audio cannot be empty');
    });

    it.skipIf(!canRunNativeTests)('throws error for invalid audio type', async () => {
      // @ts-expect-error Testing invalid input
      await expect(model.diarize([1, 2, 3])).rejects.toThrow('audio must be a Float32Array');
    });

    it.skipIf(!canRunNativeTests)('throws error for invalid threshold', async () => {
      const audio = new Float32Array(16000);
      await expect(model.diarize(audio, { threshold: -0.1 })).rejects.toThrow('threshold must be a number between 0 and 1');
      await expect(model.diarize(audio, { threshold: 1.5 })).rejects.toThrow('threshold must be a number between 0 and 1');
    });

    it.skipIf(!canRunNativeTests)('throws error for invalid medianFilter', async () => {
      const audio = new Float32Array(16000);
      await expect(model.diarize(audio, { medianFilter: 0 })).rejects.toThrow('medianFilter must be a positive odd integer');
      await expect(model.diarize(audio, { medianFilter: 4 })).rejects.toThrow('medianFilter must be a positive odd integer');
    });
  });

  describe('Latency Presets', () => {
    let model: Sortformer;

    beforeAll(async () => {
      if (canRunNativeTests) {
        model = await Sortformer.load(MODEL_PATH, { threads: 4 });
      }
    });

    afterAll(() => {
      model?.close();
    });

    const presets: LatencyPreset[] = ['low', '2s', '3s', '5s'];

    presets.forEach((preset) => {
      it.skipIf(!canRunNativeTests)(`works with latency preset: ${preset}`, async () => {
        const audio = new Float32Array(16000 * 3); // 3 seconds
        const result = await model.diarize(audio, { mode: 'streaming', latency: preset });
        
        expect(result).toBeDefined();
        expect(result.rttm).toBeDefined();
        expect(result.predictions).toBeInstanceOf(Float32Array);
      });
    });

    it.skipIf(!canRunNativeTests)('throws error for invalid latency preset', async () => {
      const audio = new Float32Array(16000);
      // @ts-expect-error Testing invalid preset
      await expect(model.diarize(audio, { mode: 'streaming', latency: 'invalid' })).rejects.toThrow('Unknown latency preset');
    });
  });

  describe('Model Lifecycle', () => {
    it.skipIf(!canRunNativeTests)('close() is idempotent', async () => {
      const model = await Sortformer.load(MODEL_PATH);
      
      expect(model.isClosed()).toBe(false);
      
      // First close
      model.close();
      expect(model.isClosed()).toBe(true);
      
      // Second close should not throw
      expect(() => model.close()).not.toThrow();
      expect(model.isClosed()).toBe(true);
      
      // Third close should still not throw
      expect(() => model.close()).not.toThrow();
    });

    it.skipIf(!canRunNativeTests)('diarize after close() throws error', async () => {
      const model = await Sortformer.load(MODEL_PATH);
      model.close();
      
      const audio = new Float32Array(16000);
      await expect(model.diarize(audio)).rejects.toThrow('Model is closed');
    });
  });

  describe('Diarization Modes', () => {
    let model: Sortformer;

    beforeAll(async () => {
      if (canRunNativeTests) {
        model = await Sortformer.load(MODEL_PATH, { threads: 4 });
      }
    });

    afterAll(() => {
      model?.close();
    });

    it.skipIf(!canRunNativeTests)('works in offline mode', async () => {
      const audio = new Float32Array(16000 * 3);
      const result = await model.diarize(audio, { mode: 'offline' });
      
      expect(result).toBeDefined();
      expect(result.rttm).toBeDefined();
    });

    it.skipIf(!canRunNativeTests)('works in streaming mode', async () => {
      const audio = new Float32Array(16000 * 3);
      const result = await model.diarize(audio, { mode: 'streaming' });
      
      expect(result).toBeDefined();
      expect(result.rttm).toBeDefined();
    });

    it.skipIf(!canRunNativeTests)('throws error for invalid mode', async () => {
      const audio = new Float32Array(16000);
      // @ts-expect-error Testing invalid mode
      await expect(model.diarize(audio, { mode: 'invalid' })).rejects.toThrow('Unknown diarization mode');
    });
  });
});
