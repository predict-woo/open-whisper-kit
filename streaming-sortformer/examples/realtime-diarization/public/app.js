// WebSocket connection
let ws = null;
let isConnected = false;

// DOM elements
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const levelFill = document.getElementById('levelFill');
const logContainer = document.getElementById('logContainer');
const timelineCanvas = document.getElementById('timelineCanvas');
const ctx = timelineCanvas.getContext('2d');
const downloadsSection = document.getElementById('downloadsSection');
const downloadInfo = document.getElementById('downloadInfo');
const downloadAudio = document.getElementById('downloadAudio');
const downloadRttm = document.getElementById('downloadRttm');
const presetSelect = document.getElementById('presetSelect');

// Speaker elements
const speakerFills = [
  document.getElementById('speaker0'),
  document.getElementById('speaker1'),
  document.getElementById('speaker2'),
  document.getElementById('speaker3')
];
const speakerValues = [
  document.getElementById('speaker0Value'),
  document.getElementById('speaker1Value'),
  document.getElementById('speaker2Value'),
  document.getElementById('speaker3Value')
];

// Speaker colors
const SPEAKER_COLORS = ['#3b82f6', '#22c55e', '#f97316', '#a855f7'];

// Timeline data (last 30 seconds, 10 frames per second = 300 frames)
const TIMELINE_DURATION = 30;
const FRAMES_PER_SECOND = 10;
const MAX_FRAMES = TIMELINE_DURATION * FRAMES_PER_SECOND;
let timelineData = [];

// Segments for timeline
let segments = [];

// Update status display
function setStatus(message, connected, ready = false) {
  statusText.textContent = message;
  statusDot.className = 'status-dot ' + (connected ? (ready ? 'connected' : 'connecting') : 'disconnected');
}

// Log an event
function log(message, type = 'info') {
  const entry = document.createElement('div');
  entry.className = `log-entry log-${type}`;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  logContainer.insertBefore(entry, logContainer.firstChild);
  
  // Keep only last 50 entries
  while (logContainer.children.length > 50) {
    logContainer.removeChild(logContainer.lastChild);
  }
}

// Update speaker bars
function updateSpeakers(speakers) {
  for (let i = 0; i < 4; i++) {
    const value = Math.round((speakers[i] || 0) * 100);
    speakerFills[i].style.width = value + '%';
    speakerFills[i].style.backgroundColor = SPEAKER_COLORS[i];
    speakerValues[i].textContent = value + '%';
  }
  
  // Add to timeline data
  timelineData.push([...speakers]);
  if (timelineData.length > MAX_FRAMES) {
    timelineData.shift();
  }
  
  drawTimeline();
}

// Update audio level
function updateLevel(level) {
  const percent = Math.round(level * 100);
  levelFill.style.width = percent + '%';
}

// Show download links
function showDownloads(downloads) {
  if (!downloads) {
    downloadsSection.style.display = 'none';
    return;
  }
  
  downloadInfo.textContent = `Duration: ${downloads.duration}s | Segments: ${downloads.segments}`;
  downloadAudio.href = downloads.audio;
  downloadAudio.download = downloads.audio.split('/').pop();
  downloadRttm.href = downloads.rttm;
  downloadRttm.download = downloads.rttm.split('/').pop();
  downloadsSection.style.display = 'block';
  
  log(`Recording saved: ${downloads.duration}s, ${downloads.segments} segments`, 'success');
}

// Add segment to timeline
function addSegment(speaker, start, duration) {
  segments.push({ speaker, start, duration, addedAt: Date.now() / 1000 });
  
  // Remove segments older than 30 seconds
  const now = Date.now() / 1000;
  segments = segments.filter(s => now - s.addedAt < TIMELINE_DURATION);
}

// Draw timeline canvas
function drawTimeline() {
  const width = timelineCanvas.width;
  const height = timelineCanvas.height;
  const rowHeight = height / 4;
  
  // Clear canvas
  ctx.fillStyle = '#1f2937';
  ctx.fillRect(0, 0, width, height);
  
  // Draw grid lines
  ctx.strokeStyle = '#374151';
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    ctx.beginPath();
    ctx.moveTo(0, i * rowHeight);
    ctx.lineTo(width, i * rowHeight);
    ctx.stroke();
  }
  
  // Draw time markers
  ctx.fillStyle = '#6b7280';
  ctx.font = '10px sans-serif';
  for (let i = 0; i <= TIMELINE_DURATION; i += 5) {
    const x = (i / TIMELINE_DURATION) * width;
    ctx.fillText(i + 's', x + 2, height - 2);
  }
  
  // Draw speaker labels
  for (let i = 0; i < 4; i++) {
    ctx.fillStyle = SPEAKER_COLORS[i];
    ctx.fillText(`Spk ${i}`, 5, i * rowHeight + 15);
  }
  
  // Draw timeline data
  if (timelineData.length > 0) {
    const frameWidth = width / MAX_FRAMES;
    
    for (let frame = 0; frame < timelineData.length; frame++) {
      const x = ((MAX_FRAMES - timelineData.length + frame) / MAX_FRAMES) * width;
      const speakers = timelineData[frame];
      
      for (let speaker = 0; speaker < 4; speaker++) {
        const value = speakers[speaker] || 0;
        if (value >= 0.5) { // Only draw if above threshold (binary on/off)
          ctx.fillStyle = SPEAKER_COLORS[speaker];
          ctx.fillRect(x, speaker * rowHeight + 2, Math.max(1, frameWidth), rowHeight - 4);
        }
      }
    }
  }
}

// Connect to WebSocket
function connect() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${window.location.host}`);
  
  ws.onopen = () => {
    isConnected = true;
    log('Connected to server', 'success');
    startBtn.disabled = false;
  };
  
  ws.onclose = () => {
    isConnected = false;
    setStatus('Disconnected', false);
    startBtn.disabled = true;
    stopBtn.disabled = true;
    log('Disconnected from server', 'error');
    
    // Reconnect after 3 seconds
    setTimeout(connect, 3000);
  };
  
  ws.onerror = (error) => {
    log('WebSocket error', 'error');
  };
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'status':
          setStatus(data.message, true, data.ready);
          if (data.recording !== undefined) {
            startBtn.disabled = data.recording;
            stopBtn.disabled = !data.recording;
          }
          if (data.ready) {
            startBtn.disabled = false;
          }
          if (data.downloads) {
            showDownloads(data.downloads);
          }
          log(data.message, 'info');
          break;
          
        case 'predictions':
          updateSpeakers(data.speakers);
          break;
          
        case 'level':
          updateLevel(data.level);
          break;
          
        case 'segment':
          addSegment(data.speaker, data.start, data.duration);
          log(`Speaker ${data.speaker}: ${data.start.toFixed(1)}s - ${(data.start + data.duration).toFixed(1)}s`, 'segment');
          break;
          
        case 'error':
          log(data.message, 'error');
          break;
      }
    } catch (err) {
      console.error('Failed to parse message:', err);
    }
  };
}

// Button handlers
startBtn.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    const preset = presetSelect.value;
    ws.send(JSON.stringify({ type: 'start', preset }));
    startBtn.disabled = true;
    stopBtn.disabled = false;
    presetSelect.disabled = true;
    downloadsSection.style.display = 'none';
    timelineData = [];
    drawTimeline();
    log(`Recording started (preset: ${preset})`, 'info');
  }
});

stopBtn.addEventListener('click', () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'stop' }));
    startBtn.disabled = false;
    stopBtn.disabled = true;
    presetSelect.disabled = false;
    log('Recording stopped', 'info');
  }
});

// Initialize
setStatus('Connecting...', false);
drawTimeline();
connect();

// Resize canvas on window resize
window.addEventListener('resize', () => {
  timelineCanvas.width = timelineCanvas.parentElement.clientWidth;
  drawTimeline();
});

// Initial canvas sizing
timelineCanvas.width = timelineCanvas.parentElement.clientWidth || 800;
