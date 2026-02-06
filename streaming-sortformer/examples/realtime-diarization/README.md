# Real-Time Speaker Diarization Demo

A web-based demo that captures microphone audio and performs real-time speaker diarization using the SortFormer model.

## Features

- Real-time microphone audio capture
- Live speaker activity visualization (up to 4 speakers)
- 30-second scrolling timeline
- Audio level meter
- Event logging

## Prerequisites

1. **sox** - Audio capture utility
   ```bash
   # macOS
   brew install sox
   
   # Ubuntu/Debian
   sudo apt-get install sox
   
   # Fedora/RHEL
   sudo dnf install sox
   ```

2. **model.gguf** - SortFormer model file
   - Must be located at the repository root: `../../model.gguf`
   - Download or convert from NeMo checkpoint using `scripts/convert_to_gguf.py`

3. **Node.js** >= 18.0.0
   - Check version: `node --version`

4. **sortformer-diarize** - Built CLI binary
   - Must be built at `../../build/sortformer-diarize`
   - Build instructions in main README.md

## Installation

```bash
cd examples/realtime-diarization
npm install
```

## Usage

```bash
npm start
```

Then open http://localhost:3000 in your browser.

1. Wait for "Model loaded" status
2. Click "Start Recording"
3. Speak into your microphone
4. Watch real-time speaker detection
5. Click "Stop Recording" when done

## How It Works

1. The server captures microphone audio using `sox` at 16kHz mono
2. Audio is buffered for 3 seconds (48,000 samples)
3. Each buffer is processed through the SortFormer model via `sortformer-diarize`
4. Speaker predictions are sent to the browser via WebSocket
5. The UI displays real-time speaker activity with a 30-second scrolling timeline

## Troubleshooting

### "Failed to start audio capture"
- Make sure `sox` is installed: `which sox`
- Grant microphone permission to your terminal app (macOS System Settings → Privacy & Security → Microphone)
- Test sox manually: `sox -d -t raw -r 16000 -e signed -b 16 -c 1 - | head -c 1000`

### "Failed to load model"
- Verify `model.gguf` exists at repository root: `ls -lh ../../model.gguf`
- Check file permissions: `chmod 644 ../../model.gguf`
- Ensure model is valid GGUF format: `file ../../model.gguf`

### "sortformer-diarize not found"
- Build the project first:
  ```bash
  cd ../..
  cmake -B build
  cmake --build build -j$(nproc)
  ```
- Verify binary exists: `ls -lh ../../build/sortformer-diarize`

### No speakers detected
- Speak louder or move closer to microphone
- Check that audio level meter shows activity (green bars)
- Verify sox is capturing audio: check server logs for "Audio capture started"
- Test with multiple speakers or conversation

### High latency
- 3-second buffering is intentional for model accuracy
- Total latency: ~3s buffer + processing time (~0.5-1s)
- Reduce buffer size in `server.js` (line 15) for lower latency but less accuracy

### WebSocket connection failed
- Check that port 3000 is not in use: `lsof -i :3000`
- Try a different port: `PORT=3001 npm start`
- Check firewall settings

## Architecture

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐
│ Microphone  │────▶│  Node.js Server      │────▶│  Web UI     │
│ (via sox)   │     │  - Audio capture     │ WS  │  - Speakers │
└─────────────┘     │  - Buffer (3s)       │────▶│  - Timeline │
                    │  - sortformer-diarize│     │  - Log      │
                    │  - WebSocket server  │     │  - Meter    │
                    └──────────────────────┘     └─────────────┘
```

## Technical Details

- **Audio format**: 16kHz, mono, 16-bit signed PCM
- **Buffer size**: 3 seconds (48,000 samples)
- **Model input**: Raw PCM audio
- **Model output**: 4 speaker probabilities per frame (10ms resolution)
- **WebSocket protocol**: JSON messages with speaker predictions
- **Frontend**: Vanilla JavaScript with Canvas API for visualization

## Performance

- **Processing time**: ~0.5-1s per 3-second buffer (CPU)
- **Memory usage**: ~200MB (model + buffers)
- **Network**: Minimal (only prediction data, ~1KB per buffer)

## Limitations

- Maximum 4 speakers supported
- 3-second latency due to buffering
- CPU-only inference (no GPU acceleration in this demo)
- No speaker identity tracking across buffers
- No audio playback (visualization only)

## Future Enhancements

- [ ] Adjustable buffer size
- [ ] Speaker identity tracking
- [ ] Audio playback with speaker labels
- [ ] Export diarization to RTTM format
- [ ] GPU acceleration support
- [ ] Multi-channel audio support
