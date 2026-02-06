import express from 'express';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

// ESM __dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import Sortformer from local bindings
const { Sortformer } = await import('../../bindings/node/packages/streaming-sortformer-node/dist/index.js');

const PORT = 3000;
const SAMPLE_RATE = 16000;

// Recordings directory
const RECORDINGS_DIR = path.join(__dirname, 'recordings');
if (!fs.existsSync(RECORDINGS_DIR)) {
  fs.mkdirSync(RECORDINGS_DIR, { recursive: true });
}

// Global state
let model = null;
let isRecording = false;
let soxProcess = null;
let totalSamples = 0;

// Recording state - keeps ALL audio and RTTM for the session
let recordingAudio = [];  // All audio samples for current recording
let recordingStartTime = null;
let currentRecordingId = null;

// Streaming session state
let streamingSession = null;
let allPredictions = [];  // Accumulate predictions for final RTTM

// Create Express app
const app = express();
app.use(express.static(path.join(__dirname, 'public')));
app.use('/recordings', express.static(RECORDINGS_DIR));

// Create HTTP server and WebSocket server
const server = createServer(app);
const wss = new WebSocketServer({ server });

// Broadcast to all connected clients
function broadcast(data) {
  const message = JSON.stringify(data);
  wss.clients.forEach(client => {
    if (client.readyState === 1) { // WebSocket.OPEN
      client.send(message);
    }
  });
}

// Convert Buffer of 16-bit PCM to Float32Array
function pcmToFloat32(buffer) {
  const int16 = new Int16Array(buffer.buffer, buffer.byteOffset, buffer.length / 2);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768.0;
  }
  return float32;
}

// Generate RTTM from predictions
function generateRttm(predictions, frameCount, filename, threshold = 0.5, medianFilter = 11) {
  const n_spk = 4;
  const frameDur = 0.08;
  
  let basename = filename;
  const slashIdx = basename.lastIndexOf('/');
  if (slashIdx !== -1) basename = basename.substring(slashIdx + 1);
  const dotIdx = basename.lastIndexOf('.');
  if (dotIdx !== -1) basename = basename.substring(0, dotIdx);
  
  const active = new Uint8Array(frameCount * n_spk);
  for (let t = 0; t < frameCount; t++) {
    for (let s = 0; s < n_spk; s++) {
      active[t * n_spk + s] = predictions[t * n_spk + s] >= threshold ? 1 : 0;
    }
  }
  
  const filtered = new Uint8Array(frameCount * n_spk);
  const half = Math.floor(medianFilter / 2);
  for (let s = 0; s < n_spk; s++) {
    for (let t = 0; t < frameCount; t++) {
      let ones = 0;
      for (let j = t - half; j <= t + half; j++) {
        if (j >= 0 && j < frameCount) {
          ones += active[j * n_spk + s];
        }
      }
      filtered[t * n_spk + s] = (ones * 2 > medianFilter) ? 1 : 0;
    }
  }
  
  const lines = [];
  for (let s = 0; s < n_spk; s++) {
    let inSegment = false;
    let segStart = 0;
    
    for (let t = 0; t <= frameCount; t++) {
      const isActive = t < frameCount ? filtered[t * n_spk + s] : 0;
      
      if (isActive && !inSegment) {
        inSegment = true;
        segStart = t;
      } else if (!isActive && inSegment) {
        inSegment = false;
        const start = segStart * frameDur;
        const duration = (t - segStart) * frameDur;
        lines.push(`SPEAKER ${basename} 1 ${start.toFixed(3)} ${duration.toFixed(3)} <NA> <NA> speaker_${s} <NA> <NA>`);
      }
    }
  }
  
  lines.sort((a, b) => {
    const startA = parseFloat(a.split(/\s+/)[3]);
    const startB = parseFloat(b.split(/\s+/)[3]);
    return startA - startB;
  });
  
  return lines;
}

// Start audio capture
function startRecording(preset = '2s') {
  if (isRecording) return;
  
  console.log(`Starting audio capture with preset: ${preset}...`);
  totalSamples = 0;
  
  // Initialize recording state
  recordingAudio = [];
  recordingStartTime = new Date();
  currentRecordingId = `recording_${Date.now()}`;
  
  // Create streaming session with selected preset
  streamingSession = model.createStreamingSession({ preset });
  allPredictions = [];
  
  // Spawn sox to capture from microphone
  soxProcess = spawn('sox', [
    '-d',                    // default audio device
    '-t', 'raw',             // raw output
    '-b', '16',              // 16-bit
    '-e', 'signed-integer',  // signed integer
    '-r', String(SAMPLE_RATE), // sample rate
    '-c', '1',               // mono
    '-'                      // stdout
  ]);
  
  soxProcess.stdout.on('data', (chunk) => {
    const float32 = pcmToFloat32(chunk);
    totalSamples += float32.length;
    
    recordingAudio.push(new Float32Array(float32));
    
    // Calculate audio level for visualization
    let sum = 0;
    for (let i = 0; i < float32.length; i++) {
      sum += float32[i] * float32[i];
    }
    const level = Math.sqrt(sum / float32.length);
    broadcast({ type: 'level', level: Math.min(1, level * 10) });
    
    // Feed to streaming session immediately
    if (streamingSession) {
      try {
        const result = streamingSession.feed(float32);
        
        if (result.frameCount > 0) {
          // Store predictions for final RTTM
          allPredictions.push(result.predictions);
          
          // Broadcast last frame for real-time visualization
          const lastFrameStart = (result.frameCount - 1) * 4;
          const speakers = [
            result.predictions[lastFrameStart],
            result.predictions[lastFrameStart + 1],
            result.predictions[lastFrameStart + 2],
            result.predictions[lastFrameStart + 3]
          ];
          
          const timestamp = totalSamples / SAMPLE_RATE;
          broadcast({ type: 'predictions', speakers, timestamp });
        }
      } catch (err) {
        console.error('Stream feed error:', err);
        broadcast({ type: 'error', message: err.message });
      }
    }
  });
  
  soxProcess.stderr.on('data', (data) => {
    // Sox outputs info to stderr, usually not errors
    console.log('sox:', data.toString());
  });
  
  soxProcess.on('error', (err) => {
    console.error('Sox error:', err);
    broadcast({ type: 'error', message: 'Failed to start audio capture. Is sox installed?' });
    isRecording = false;
  });
  
  soxProcess.on('close', (code) => {
    console.log('Sox process closed with code:', code);
    isRecording = false;
  });
  
  isRecording = true;
  broadcast({ type: 'status', message: 'Recording started', recording: true });
}

// Write WAV file from Float32Array samples
function writeWavFile(filepath, samples, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const fileSize = 36 + dataSize;
  
  const buffer = Buffer.alloc(44 + dataSize);
  let offset = 0;
  
  buffer.write('RIFF', offset); offset += 4;
  buffer.writeUInt32LE(fileSize, offset); offset += 4;
  buffer.write('WAVE', offset); offset += 4;
  buffer.write('fmt ', offset); offset += 4;
  buffer.writeUInt32LE(16, offset); offset += 4;
  buffer.writeUInt16LE(1, offset); offset += 2;
  buffer.writeUInt16LE(numChannels, offset); offset += 2;
  buffer.writeUInt32LE(sampleRate, offset); offset += 4;
  buffer.writeUInt32LE(byteRate, offset); offset += 4;
  buffer.writeUInt16LE(blockAlign, offset); offset += 2;
  buffer.writeUInt16LE(bitsPerSample, offset); offset += 2;
  buffer.write('data', offset); offset += 4;
  buffer.writeUInt32LE(dataSize, offset); offset += 4;
  
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    buffer.writeInt16LE(Math.round(s * 32767), offset);
    offset += 2;
  }
  
  fs.writeFileSync(filepath, buffer);
}

// Stop audio capture and save files
function stopRecording() {
  if (!isRecording) return;
  
  console.log('Stopping audio capture...');
  if (soxProcess) {
    soxProcess.kill();
    soxProcess = null;
  }
  isRecording = false;
  
  let downloadUrls = null;
  
  if (currentRecordingId && recordingAudio.length > 0) {
    const totalLength = recordingAudio.reduce((sum, buf) => sum + buf.length, 0);
    const allSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of recordingAudio) {
      allSamples.set(chunk, offset);
      offset += chunk.length;
    }
    
    const wavFilename = `${currentRecordingId}.wav`;
    const rttmFilename = `${currentRecordingId}.rttm`;
    const wavPath = path.join(RECORDINGS_DIR, wavFilename);
    const rttmPath = path.join(RECORDINGS_DIR, rttmFilename);
    
    writeWavFile(wavPath, allSamples, SAMPLE_RATE);
    console.log(`Saved audio: ${wavPath} (${(totalLength / SAMPLE_RATE).toFixed(1)}s)`);
    
    // Flush remaining buffered audio before generating RTTM
    if (streamingSession) {
      try {
        const flushResult = streamingSession.flush();
        if (flushResult.frameCount > 0) {
          allPredictions.push(flushResult.predictions);
          console.log(`Flushed ${flushResult.frameCount} final frames`);
        }
      } catch (err) {
        console.error('Flush error:', err);
      }
    }
    
    // Convert accumulated predictions to RTTM
    if (allPredictions.length > 0) {
      const totalPredLength = allPredictions.reduce((sum, p) => sum + p.length, 0);
      const allPreds = new Float32Array(totalPredLength);
      let predOffset = 0;
      for (const preds of allPredictions) {
        allPreds.set(preds, predOffset);
        predOffset += preds.length;
      }
      
      const totalFrames = allPreds.length / 4;
      console.log(`Total frames: ${totalFrames}`);
      
      const rttmLines = generateRttm(allPreds, totalFrames, wavFilename, 0.5, 11);
      fs.writeFileSync(rttmPath, rttmLines.join('\n') + '\n');
      console.log(`Saved RTTM: ${rttmPath} (${rttmLines.length} segments)`);
      
      downloadUrls = {
        audio: `/recordings/${wavFilename}`,
        rttm: `/recordings/${rttmFilename}`,
        duration: (totalLength / SAMPLE_RATE).toFixed(1),
        segments: rttmLines.length
      };
    }
  }
  
  totalSamples = 0;
  recordingAudio = [];
  currentRecordingId = null;
  
  if (streamingSession) {
    streamingSession.close();
    streamingSession = null;
  }
  allPredictions = [];
  
  broadcast({ 
    type: 'status', 
    message: 'Recording stopped', 
    recording: false,
    downloads: downloadUrls
  });
}

// Handle WebSocket connections
wss.on('connection', (ws) => {
  console.log('Client connected');
  
  // Send initial status
  ws.send(JSON.stringify({
    type: 'status',
    message: model ? 'Ready' : 'Loading model...',
    ready: !!model,
    recording: isRecording
  }));
  
  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data);
      if (msg.type === 'start') {
        const preset = msg.preset || '2s';
        startRecording(preset);
      } else if (msg.type === 'stop') {
        stopRecording();
      }
    } catch (err) {
      console.error('Invalid message:', err);
    }
  });
  
  ws.on('close', () => {
    console.log('Client disconnected');
  });
});

// Initialize
async function init() {
  console.log('Loading model...');
  
  // Model path - look in repo root
  const modelPath = path.resolve(__dirname, '../../model.gguf');
  
  try {
    model = await Sortformer.load(modelPath, { threads: 4 });
    console.log('Model loaded successfully');
    broadcast({ type: 'status', message: 'Model loaded', ready: true });
  } catch (err) {
    console.error('Failed to load model:', err);
    broadcast({ type: 'error', message: `Failed to load model: ${err.message}` });
  }
  
  server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
  });
}

// Handle shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  stopRecording();
  if (model) model.close();
  process.exit(0);
});

init();
