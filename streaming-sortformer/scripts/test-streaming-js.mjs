#!/usr/bin/env node
/**
 * Test script to compare Node.js streaming API output with C++ CLI output
 * 
 * This script processes test.wav through the Node.js streaming API in chunks
 * (simulating real-time microphone input) and compares the resulting RTTM
 * with the C++ CLI's output.
 * 
 * Usage:
 *   node scripts/test-streaming-js.mjs [--chunk-size <samples>] [--preset <preset>]
 * 
 * Default chunk size simulates sox output (~2048 samples per chunk)
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Constants
const SAMPLE_RATE = 16000;
const FRAME_DUR = 0.08;  // 80ms per frame
const N_SPK = 4;

// Parse command line arguments
const args = process.argv.slice(2);
let chunkSize = 2048;  // Default: typical sox chunk size
let preset = '2s';
let wavPath = path.resolve(__dirname, '../test.wav');

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--chunk-size' && args[i + 1]) {
    chunkSize = parseInt(args[i + 1]);
    i++;
  } else if (args[i] === '--preset' && args[i + 1]) {
    preset = args[i + 1];
    i++;
  } else if (args[i] === '--wav' && args[i + 1]) {
    wavPath = path.resolve(args[i + 1]);
    i++;
  }
}

/**
 * Load WAV file and return Float32Array of samples
 */
function loadWav(filepath) {
  const buffer = fs.readFileSync(filepath);
  
  // Parse WAV header
  const riff = buffer.toString('ascii', 0, 4);
  if (riff !== 'RIFF') {
    throw new Error('Not a RIFF file');
  }
  
  const wave = buffer.toString('ascii', 8, 12);
  if (wave !== 'WAVE') {
    throw new Error('Not a WAVE file');
  }
  
  // Find fmt and data chunks
  let offset = 12;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let numChannels = 0;
  let dataStart = 0;
  let dataSize = 0;
  
  while (offset < buffer.length) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    
    if (chunkId === 'fmt ') {
      numChannels = buffer.readUInt16LE(offset + 10);
      sampleRate = buffer.readUInt32LE(offset + 12);
      bitsPerSample = buffer.readUInt16LE(offset + 22);
    } else if (chunkId === 'data') {
      dataStart = offset + 8;
      dataSize = chunkSize;
      break;
    }
    
    offset += 8 + chunkSize;
  }
  
  if (sampleRate !== 16000) {
    throw new Error(`Expected 16kHz, got ${sampleRate}Hz`);
  }
  if (numChannels !== 1) {
    throw new Error(`Expected mono, got ${numChannels} channels`);
  }
  if (bitsPerSample !== 16) {
    throw new Error(`Expected 16-bit, got ${bitsPerSample}-bit`);
  }
  
  // Convert to Float32Array
  const numSamples = dataSize / 2;
  const samples = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    samples[i] = buffer.readInt16LE(dataStart + i * 2) / 32768.0;
  }
  
  return samples;
}

/**
 * Generate RTTM from predictions
 */
function generateRttm(predictions, frameCount, filename, threshold = 0.5, medianFilter = 11) {
  let basename = path.basename(filename, path.extname(filename));
  
  // Threshold predictions
  const active = new Uint8Array(frameCount * N_SPK);
  for (let t = 0; t < frameCount; t++) {
    for (let s = 0; s < N_SPK; s++) {
      active[t * N_SPK + s] = predictions[t * N_SPK + s] >= threshold ? 1 : 0;
    }
  }
  
  // Apply median filter
  const filtered = new Uint8Array(frameCount * N_SPK);
  const half = Math.floor(medianFilter / 2);
  for (let s = 0; s < N_SPK; s++) {
    for (let t = 0; t < frameCount; t++) {
      let ones = 0;
      for (let j = t - half; j <= t + half; j++) {
        if (j >= 0 && j < frameCount) {
          ones += active[j * N_SPK + s];
        }
      }
      filtered[t * N_SPK + s] = (ones * 2 > medianFilter) ? 1 : 0;
    }
  }
  
  // Generate RTTM lines
  const lines = [];
  for (let s = 0; s < N_SPK; s++) {
    let inSegment = false;
    let segStart = 0;
    
    for (let t = 0; t <= frameCount; t++) {
      const isActive = t < frameCount ? filtered[t * N_SPK + s] : 0;
      
      if (isActive && !inSegment) {
        inSegment = true;
        segStart = t;
      } else if (!isActive && inSegment) {
        inSegment = false;
        const start = segStart * FRAME_DUR;
        const duration = (t - segStart) * FRAME_DUR;
        lines.push(`SPEAKER ${basename} 1 ${start.toFixed(3)} ${duration.toFixed(3)} <NA> <NA> speaker_${s} <NA> <NA>`);
      }
    }
  }
  
  // Sort by start time
  lines.sort((a, b) => {
    const startA = parseFloat(a.split(/\s+/)[3]);
    const startB = parseFloat(b.split(/\s+/)[3]);
    return startA - startB;
  });
  
  return lines;
}

async function main() {
  console.log('=== Node.js Streaming API Test ===\n');
  
  // Load Sortformer
  console.log('Loading Sortformer...');
  const { Sortformer } = await import('../bindings/node/packages/streaming-sortformer-node/dist/index.js');
  
  const modelPath = path.resolve(__dirname, '../model.gguf');
  const model = await Sortformer.load(modelPath, { threads: 4 });
  console.log('Model loaded\n');
  
  // Load WAV file
  console.log(`Loading WAV: ${wavPath}`);
  const samples = loadWav(wavPath);
  const duration = samples.length / SAMPLE_RATE;
  console.log(`  Duration: ${duration.toFixed(2)}s`);
  console.log(`  Samples: ${samples.length}`);
  console.log(`  Expected frames: ${Math.floor(samples.length / 160 / 8)}`);
  console.log();
  
  // Create streaming session
  console.log(`Creating streaming session with preset: ${preset}`);
  const session = model.createStreamingSession({ preset });
  
  // Process in chunks (simulating real-time audio input)
  console.log(`Processing in chunks of ${chunkSize} samples (${(chunkSize / SAMPLE_RATE * 1000).toFixed(1)}ms)...\n`);
  
  const allPredictions = [];
  let totalFrames = 0;
  let chunkCount = 0;
  
  for (let i = 0; i < samples.length; i += chunkSize) {
    const chunk = samples.slice(i, Math.min(i + chunkSize, samples.length));
    const result = session.feed(chunk);
    
    if (result.frameCount > 0) {
      allPredictions.push(result.predictions);
      totalFrames += result.frameCount;
    }
    chunkCount++;
    
    // Print progress every 10 chunks
    if (chunkCount % 100 === 0) {
      const progress = ((i + chunkSize) / samples.length * 100).toFixed(1);
      const audioTime = ((i + chunkSize) / SAMPLE_RATE).toFixed(2);
      const rttmTime = (totalFrames * FRAME_DUR).toFixed(2);
      console.log(`  Chunk ${chunkCount}: audio=${audioTime}s, frames=${totalFrames}, rttm_time=${rttmTime}s`);
    }
  }
  
  // Flush remaining buffered audio
  console.log('\nFlushing remaining audio...');
  const flushResult = session.flush();
  if (flushResult.frameCount > 0) {
    allPredictions.push(flushResult.predictions);
    totalFrames += flushResult.frameCount;
    console.log(`  Flushed ${flushResult.frameCount} additional frames`);
  }
  
  session.close();
  model.close();
  
  // Combine all predictions
  const totalPredLength = allPredictions.reduce((sum, p) => sum + p.length, 0);
  const combinedPreds = new Float32Array(totalPredLength);
  let predOffset = 0;
  for (const preds of allPredictions) {
    combinedPreds.set(preds, predOffset);
    predOffset += preds.length;
  }
  
  // Print summary
  console.log('\n=== Results ===\n');
  console.log(`Audio duration: ${duration.toFixed(3)}s`);
  console.log(`Total chunks processed: ${chunkCount}`);
  console.log(`Total frames output: ${totalFrames}`);
  console.log(`RTTM duration: ${(totalFrames * FRAME_DUR).toFixed(3)}s`);
  console.log(`Drift: ${((totalFrames * FRAME_DUR) - duration).toFixed(3)}s`);
  
  // Calculate expected frames
  // Frame count = audio_samples / hop_length / subsampling_factor
  // But this is approximate - exact count depends on padding
  const expectedFrames = Math.floor(samples.length / 1280);  // 160 * 8 = 1280 samples per frame
  console.log(`Expected frames: ~${expectedFrames}`);
  console.log(`Frame difference: ${totalFrames - expectedFrames}`);
  
  // Generate RTTM
  const rttmLines = generateRttm(combinedPreds, totalFrames, wavPath, 0.5, 11);
  
  // Write RTTM to file
  const outPath = path.resolve(__dirname, '../js_streaming.rttm');
  fs.writeFileSync(outPath, rttmLines.join('\n') + '\n');
  console.log(`\nRTTM saved to: ${outPath}`);
  console.log(`  Segments: ${rttmLines.length}`);
  
  // Compare with C++ output if available
  const cppRttmPath = path.resolve(__dirname, '../output.rttm');
  if (fs.existsSync(cppRttmPath)) {
    console.log('\n=== Comparison with C++ CLI output ===\n');
    const cppRttm = fs.readFileSync(cppRttmPath, 'utf8').trim().split('\n');
    console.log(`C++ RTTM segments: ${cppRttm.length}`);
    console.log(`JS RTTM segments: ${rttmLines.length}`);
    
    // Compare last segment end times
    const cppLast = cppRttm[cppRttm.length - 1];
    const jsLast = rttmLines[rttmLines.length - 1];
    
    const cppEndTime = parseFloat(cppLast.split(/\s+/)[3]) + parseFloat(cppLast.split(/\s+/)[4]);
    const jsEndTime = parseFloat(jsLast.split(/\s+/)[3]) + parseFloat(jsLast.split(/\s+/)[4]);
    
    console.log(`C++ last segment end: ${cppEndTime.toFixed(3)}s`);
    console.log(`JS last segment end: ${jsEndTime.toFixed(3)}s`);
    console.log(`End time difference: ${(jsEndTime - cppEndTime).toFixed(3)}s`);
  }
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
