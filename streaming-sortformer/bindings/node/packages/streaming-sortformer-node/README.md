# streaming-sortformer-node

Node.js bindings for SortFormer streaming speaker diarization.

## Installation

```bash
npm install streaming-sortformer-node
```

Supported platforms:
- macOS Apple Silicon (arm64)
- macOS Intel (x64)

## Quick Start

```javascript
const { Sortformer } = require('streaming-sortformer-node');

// Load model
const model = await Sortformer.load('./model.gguf', { threads: 4 });

// Prepare audio (16kHz mono Float32Array)
const audio = new Float32Array(/* your audio samples */);

// Run diarization
const result = await model.diarize(audio, {
  mode: 'streaming',
  latency: '2s',
  threshold: 0.5
});

// Output RTTM format
console.log(result.rttm);

// Access raw predictions
console.log(`Detected ${result.speakerCount} speakers`);
console.log(`Frame count: ${result.frameCount}`);

// Clean up
model.close();
```

## API Reference

### `Sortformer.load(modelPath, options?)`

Load a SortFormer model from a GGUF file.

**Parameters:**
- `modelPath` (string): Path to the GGUF model file
- `options` (LoadOptions, optional):
  - `threads` (number): Number of CPU threads for inference (default: 4)

**Returns:** `Promise<Sortformer>`

**Example:**
```javascript
const model = await Sortformer.load('./model.gguf', { threads: 8 });
```

### `model.diarize(audio, options?)`

Run diarization inference on audio samples.

**Parameters:**
- `audio` (Float32Array): Audio samples at 16kHz mono
- `options` (DiarizeOptions, optional):
  - `mode` ('offline' | 'streaming'): Diarization mode (default: 'offline')
  - `latency` ('low' | '2s' | '3s' | '5s'): Latency preset for streaming mode (default: '2s')
  - `threshold` (number): Speaker activity threshold, 0.0-1.0 (default: 0.5)
  - `medianFilter` (number): Median filter window size, must be odd (default: 11)

**Returns:** `Promise<DiarizeResult>`

**DiarizeResult:**
- `rttm` (string): RTTM format output with speaker segments
- `predictions` (Float32Array): Raw per-frame predictions, shape [frameCount, 4]
- `frameCount` (number): Number of frames in output
- `speakerCount` (number): Number of speakers detected (1-4)

**Example:**
```javascript
const result = await model.diarize(audio, {
  mode: 'streaming',
  latency: '2s',
  threshold: 0.5,
  medianFilter: 11
});
```

### `model.close()`

Close the model and free native resources.

After calling `close()`, the model cannot be used for further inference. Calling `close()` multiple times is safe (idempotent).

**Example:**
```javascript
model.close();
```

### `model.isClosed()`

Check if the model has been closed.

**Returns:** `boolean` - true if closed, false otherwise

## Types

### `LoadOptions`

```typescript
interface LoadOptions {
  threads?: number;  // Number of CPU threads (default: auto-detected)
}
```

### `DiarizeOptions`

```typescript
interface DiarizeOptions {
  mode?: 'offline' | 'streaming';  // Default: 'offline'
  latency?: 'low' | '2s' | '3s' | '5s';  // Default: '2s' (streaming only)
  threshold?: number;  // 0.0-1.0, default: 0.5
  medianFilter?: number;  // Odd integer >= 1, default: 11
}
```

### `DiarizeResult`

```typescript
interface DiarizeResult {
  rttm: string;  // RTTM format output
  predictions: Float32Array;  // Shape: [frameCount, 4]
  frameCount: number;  // Number of frames
  speakerCount: number;  // Detected speakers (1-4)
}
```

## Latency Presets

Streaming mode supports four latency presets that trade off latency for accuracy:

| Preset | Latency | Chunk Size | Use Case |
|--------|---------|------------|----------|
| `low` | ~188ms | 6 frames | Real-time applications, minimal delay |
| `2s` | ~2 seconds | 15 frames | Near real-time, balanced accuracy |
| `3s` | ~3 seconds | 30 frames | Higher accuracy, moderate delay |
| `5s` | ~5 seconds | 55 frames | Best accuracy, higher latency acceptable |

**Offline mode** processes the entire audio at once with no streaming constraints, providing the highest accuracy but requiring the full audio upfront.

## Audio Format

Input audio must be:
- **Sample rate:** 16kHz
- **Channels:** Mono (single channel)
- **Format:** Float32Array with values in range [-1.0, 1.0]

To convert from 16-bit PCM:

```javascript
// From Int16Array
const float32 = new Float32Array(int16Array.length);
for (let i = 0; i < int16Array.length; i++) {
  float32[i] = int16Array[i] / 32768.0;
}

// From Buffer (16-bit PCM)
const int16 = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.length / 2);
const float32 = new Float32Array(int16.length);
for (let i = 0; i < int16.length; i++) {
  float32[i] = int16[i] / 32768.0;
}
```

## CoreML Acceleration

On Apple Silicon Macs, the addon automatically uses CoreML/ANE acceleration if a compiled CoreML model is present alongside the GGUF file.

**Setup:**

1. Convert the model head to CoreML:
```bash
python scripts/convert_head_to_coreml.py \
  --model model.nemo \
  --output model-coreml-head.mlpackage \
  --precision fp16
```

2. Compile the CoreML model:
```bash
xcrun coremlcompiler compile model-coreml-head.mlpackage .
```

3. Place `model-coreml-head.mlmodelc/` in the same directory as `model.gguf`

The addon will automatically detect and use the CoreML model, providing ~110x real-time performance on Apple Silicon (vs ~10x with CPU-only).

**Performance comparison (M3 MacBook Pro):**

| Backend | Speed | Memory |
|---------|-------|--------|
| CoreML/ANE | ~110x real-time | ~300 MB |
| CPU only | ~10x real-time | ~380 MB |

## RTTM Output Format

The `rttm` field in `DiarizeResult` follows the standard [RTTM format](https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf):

```
SPEAKER <filename> 1 <start> <duration> <NA> <NA> speaker_<id> <NA> <NA>
```

Each line represents a contiguous speech segment for one speaker.

**Example:**
```
SPEAKER audio 1 0.000 2.560 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER audio 1 2.560 1.920 <NA> <NA> speaker_1 <NA> <NA>
SPEAKER audio 1 4.480 3.200 <NA> <NA> speaker_0 <NA> <NA>
```

## Troubleshooting

### Model not found

**Error:** `Error: Failed to load model: model.gguf`

**Solution:** Ensure the model path is correct and the file exists. Use absolute paths if relative paths fail:

```javascript
const path = require('path');
const modelPath = path.resolve(__dirname, 'model.gguf');
const model = await Sortformer.load(modelPath);
```

### Unsupported platform

**Error:** `Error: Platform not supported`

**Solution:** The addon currently only supports macOS (arm64 and x64). Windows and Linux support is planned for future releases.

### Out of memory

**Error:** `Error: Failed to allocate memory`

**Solution:** Reduce the number of threads or process shorter audio segments:

```javascript
// Reduce threads
const model = await Sortformer.load('./model.gguf', { threads: 2 });

// Process in chunks
const chunkSize = 16000 * 30; // 30 seconds at 16kHz
for (let i = 0; i < audio.length; i += chunkSize) {
  const chunk = audio.slice(i, i + chunkSize);
  const result = await model.diarize(chunk);
  // Process result...
}
```

### Invalid audio format

**Error:** `TypeError: audio must be a Float32Array`

**Solution:** Ensure audio is a Float32Array with 16kHz mono samples:

```javascript
// Check audio format
if (!(audio instanceof Float32Array)) {
  throw new Error('Audio must be Float32Array');
}

// Check sample rate (if you have metadata)
if (sampleRate !== 16000) {
  // Resample to 16kHz using a library like 'audio-resampler'
}
```

### CoreML model not loading

**Error:** Model loads but doesn't use CoreML acceleration

**Solution:**
1. Verify `model-coreml-head.mlmodelc/` is in the same directory as `model.gguf`
2. Ensure the CoreML model was compiled with `xcrun coremlcompiler compile`
3. Check that you're running on Apple Silicon (CoreML is not available on Intel Macs)

## Examples

### Real-time streaming from microphone

```javascript
const { Sortformer } = require('streaming-sortformer-node');
const mic = require('mic');

const model = await Sortformer.load('./model.gguf');

const micInstance = mic({
  rate: '16000',
  channels: '1',
  encoding: 'signed-integer',
  bitwidth: '16'
});

const micInputStream = micInstance.getAudioStream();
let buffer = [];

micInputStream.on('data', async (chunk) => {
  // Convert to Float32Array
  const int16 = new Int16Array(chunk.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768.0;
  }
  
  buffer.push(float32);
  
  // Process every 2 seconds
  if (buffer.length >= 10) {
    const audio = Float32Array.from(buffer.flat());
    const result = await model.diarize(audio, {
      mode: 'streaming',
      latency: 'low'
    });
    console.log(result.rttm);
    buffer = [];
  }
});

micInstance.start();
```

### Batch processing multiple files

```javascript
const { Sortformer } = require('streaming-sortformer-node');
const fs = require('fs');
const path = require('path');

const model = await Sortformer.load('./model.gguf', { threads: 8 });

const files = fs.readdirSync('./audio').filter(f => f.endsWith('.wav'));

for (const file of files) {
  const audio = loadAudioFile(path.join('./audio', file));
  const result = await model.diarize(audio, { mode: 'offline' });
  
  const outPath = path.join('./output', file.replace('.wav', '.rttm'));
  fs.writeFileSync(outPath, result.rttm);
  
  console.log(`Processed ${file}: ${result.speakerCount} speakers`);
}

model.close();
```

## License

This project follows the same license as the parent [streaming-sortformer-ggml](https://github.com/predict-woo/streaming-sortformer-ggml) repository.

## Links

- [GitHub Repository](https://github.com/predict-woo/streaming-sortformer-ggml)
- [Full Documentation](https://github.com/predict-woo/streaming-sortformer-ggml/tree/main/bindings/node)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [GGML](https://github.com/ggml-org/ggml)
