/**
 * Personal ARCTIC Recorder — Client-Side Application
 *
 * Implements:
 * - Lossless PCM audio capture via AudioWorklet
 * - Immediate IndexedDB durable local staging
 * - Automatic background upload with idempotency preservation
 * - Writer lease heartbeat to prevent multi-tab conflicts
 * - Strict state machine and keyboard controls
 */

(() => {
  // State variables
  let audioContext = null;
  let workletNode = null;
  let micStream = null;
  let recordingChunks = [];
  let recordingStartTime = 0;
  let timerInterval = null;
  let isRecording = false;
  let isPlaying = false;
  let micEnabled = false;

  let clientId = 'client_' + Math.random().toString(36).substring(2, 11);
  let sessionId = 'sess_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);
  let leaseInterval = null;
  let leaseAcquired = false;

  let currentPrompt = null;
  let nextPrompt = null;
  let catalogPrompts = [];
  let activeTake = null; // Most recent take for current sentence
  const RERECORD_QUEUE_KEY = 'personal-arctic-rerecord-queue';
  let rerecordMode = new URLSearchParams(window.location.search).get('rerecord_queue') === '1';
  let rerecordQueue = [];
  let rerecordFreshTakeId = null;

  const MAX_RECORDING_SECONDS = 60;
  const DB_NAME = 'PersonalArcticRecorderDB';
  const DB_STORE = 'takesQueue';

  // DOM Elements
  const btnMicEnable = document.getElementById('btn-mic-enable');
  const micSelect = document.getElementById('mic-select');
  const audioSettingsLabel = document.getElementById('audio-settings-label');
  const btnPauseSession = document.getElementById('btn-pause-session');
  const alertBanner = document.getElementById('alert-banner');
  const alertMessage = document.getElementById('alert-message');
  const btnDismissAlert = document.getElementById('btn-dismiss-alert');

  const speakerBadge = document.getElementById('speaker-badge');
  const splitBadge = document.getElementById('split-badge');
  const leaseBadge = document.getElementById('lease-badge');
  const reviewBadge = document.getElementById('review-badge');

  const keptCountEl = document.getElementById('kept-count');
  const pendingCountEl = document.getElementById('pending-count');
  const serverTakesCountEl = document.getElementById('server-takes-count');

  const currentSentenceIdEl = document.getElementById('current-sentence-id');
  const currentSplitTagEl = document.getElementById('current-split-tag');
  const currentPromptTextEl = document.getElementById('current-prompt-text');
  const nextPromptPreviewEl = document.getElementById('next-prompt-preview');

  const statusIndicator = document.getElementById('status-indicator');
  const statusText = document.getElementById('status-text');
  const timerDisplay = document.getElementById('timer-display');
  const meterFill = document.getElementById('meter-fill');

  const btnRecord = document.getElementById('btn-record');
  const btnKeep = document.getElementById('btn-keep');
  const btnRetake = document.getElementById('btn-retake');
  const btnPlayback = document.getElementById('btn-playback');

  const btnPrev = document.getElementById('btn-prev');
  const btnNext = document.getElementById('btn-next');
  const chkSkipKept = document.getElementById('chk-skip-kept');
  const inputJump = document.getElementById('input-jump');
  const btnJump = document.getElementById('btn-jump');

  const takesListEl = document.getElementById('takes-list');
  const takesSummaryEl = document.getElementById('takes-status-summary');
  const playbackAudio = document.getElementById('playback-audio');

  // -------------------------------------------------------------------------
  // IndexedDB Utilities
  // -------------------------------------------------------------------------

  function openDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1);
      request.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(DB_STORE)) {
          const store = db.createObjectStore(DB_STORE, { keyPath: 'take_id' });
          store.createIndex('status', 'status', { unique: false });
          store.createIndex('sentence_id', 'sentence_id', { unique: false });
        }
      };
      request.onsuccess = (e) => resolve(e.target.result);
      request.onerror = (e) => reject(e.target.error);
    });
  }

  async function idbPutTake(takeData) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, 'readwrite');
      const store = tx.objectStore(DB_STORE);
      const req = store.put(takeData);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  }

  async function idbGetTake(takeId) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, 'readonly');
      const store = tx.objectStore(DB_STORE);
      const req = store.get(takeId);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async function idbGetPendingTakes() {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, 'readonly');
      const store = tx.objectStore(DB_STORE);
      const req = store.getAll();
      req.onsuccess = () => {
        const all = req.result || [];
        resolve(all.filter(t => t.status === 'pending_upload' || t.status === 'pending_keep'));
      };
      req.onerror = () => reject(req.error);
    });
  }

  async function idbDeleteTake(takeId) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, 'readwrite');
      const store = tx.objectStore(DB_STORE);
      const req = store.delete(takeId);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  }

  async function idbMarkUploadComplete(takeId) {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, 'readwrite');
      const store = tx.objectStore(DB_STORE);
      const getReq = store.get(takeId);

      getReq.onerror = () => reject(getReq.error);
      getReq.onsuccess = () => {
        const latest = getReq.result;
        if (!latest) {
          resolve(null);
          return;
        }

        // Re-read inside the write transaction so an Enter press that happened
        // while the network upload was in flight is never overwritten.
        latest.status = latest.keep_decision ? 'pending_keep' : 'uploaded';
        const putReq = store.put(latest);
        putReq.onsuccess = () => resolve(latest);
        putReq.onerror = () => reject(putReq.error);
      };
    });
  }

  async function updatePendingCount() {
    try {
      const pending = await idbGetPendingTakes();
      pendingCountEl.textContent = pending.length.toString();
    } catch (err) {
      console.warn('Failed to read pending count from IDB:', err);
    }
  }

  // -------------------------------------------------------------------------
  // UI & Alerts
  // -------------------------------------------------------------------------

  function showAlert(msg, isWarning = false) {
    alertMessage.textContent = msg;
    alertBanner.className = 'alert-banner visible' + (isWarning ? ' warning' : '');
  }

  function hideAlert() {
    alertBanner.className = 'alert-banner';
    alertMessage.textContent = '';
  }

  btnDismissAlert.addEventListener('click', hideAlert);

  function setStatus(text, stateClass = '') {
    statusText.textContent = text;
    statusIndicator.className = 'status-indicator ' + stateClass;
  }

  function updateTimer(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    const formatted = `${String(mins).padStart(2, '0')}:${secs.padStart(4, '0')}`;
    timerDisplay.textContent = `${formatted} / 01:00`;
  }

  // -------------------------------------------------------------------------
  // Audio & Worklet Setup
  // -------------------------------------------------------------------------

  async function initMicrophone() {
    try {
      if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }

      await audioContext.audioWorklet.addModule('/static/capture-worklet.js');

      const selectedDeviceId = micSelect.value || undefined;
      const constraints = {
        audio: {
          deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      };

      if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
      }

      micStream = await navigator.mediaDevices.getUserMedia(constraints);
      const track = micStream.getAudioTracks()[0];
      const settings = track.getSettings ? track.getSettings() : {};

      audioSettingsLabel.textContent = `Rate: ${audioContext.sampleRate} Hz`;

      const sourceNode = audioContext.createMediaStreamSource(micStream);
      workletNode = new AudioWorkletNode(audioContext, 'capture-worklet-processor');

      workletNode.port.onmessage = (e) => {
        if (e.data.type === 'meter') {
          updateVUMeter(e.data.rms);
        } else if (e.data.type === 'pcm_chunk') {
          if (isRecording) {
            recordingChunks.push(e.data.samples);
          }
        }
      };

      sourceNode.connect(workletNode);
      // Connect to a silent dummy gain to keep the graph active without feedback
      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      workletNode.connect(silentGain);
      silentGain.connect(audioContext.destination);

      await enumerateDevices();
      micEnabled = true;
      btnMicEnable.textContent = 'Microphone Ready';
      btnMicEnable.disabled = true;
      micSelect.disabled = false;
      btnRecord.disabled = !leaseAcquired;

      setStatus('Ready');
      hideAlert();

      // Register session with backend
      await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          user_agent: navigator.userAgent,
          mic_label: track.label,
          mic_settings: settings,
        }),
      });
    } catch (err) {
      console.error('Failed to init microphone:', err);
      showAlert(`Microphone error: ${err.message || err}`);
      setStatus('Mic Error', 'error');
    }
  }

  async function enumerateDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter(d => d.kind === 'audioinput');
      const currentVal = micSelect.value;
      micSelect.innerHTML = '';

      audioInputs.forEach((dev, idx) => {
        const opt = document.createElement('option');
        opt.value = dev.deviceId;
        opt.textContent = dev.label || `Microphone ${idx + 1}`;
        micSelect.appendChild(opt);
      });

      if (currentVal && audioInputs.some(d => d.deviceId === currentVal)) {
        micSelect.value = currentVal;
      }
    } catch (err) {
      console.warn('Could not enumerate devices:', err);
    }
  }

  micSelect.addEventListener('change', async () => {
    if (micEnabled) {
      await initMicrophone();
    }
  });

  btnMicEnable.addEventListener('click', () => {
    initMicrophone();
  });

  function updateVUMeter(rms) {
    if (!meterFill) return;
    // Map RMS (0 to ~0.5) to percentage (0% to 100%)
    const level = Math.min(100, Math.max(0, rms * 250));
    meterFill.style.width = `${level}%`;
  }

  // -------------------------------------------------------------------------
  // Recording Workflow & WAV Encoding
  // -------------------------------------------------------------------------

  function startRecording() {
    if (!micEnabled || isRecording || isPlaying || !leaseAcquired) return;

    recordingChunks = [];
    isRecording = true;
    recordingStartTime = performance.now();

    // Visual lead-in (200ms)
    setStatus('Recording', 'recording');
    btnRecord.querySelector('span').textContent = 'Stop Recording';
    btnKeep.disabled = true;
    btnRetake.disabled = true;
    btnPlayback.disabled = true;
    btnPrev.disabled = true;
    btnNext.disabled = true;

    // Send start command to worklet
    workletNode.port.postMessage({ command: 'start' });

    clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      const elapsed = (performance.now() - recordingStartTime) / 1000;
      updateTimer(elapsed);
      if (elapsed >= MAX_RECORDING_SECONDS) {
        stopRecording(true);
      }
    }, 100);
  }

  async function stopRecording(hitCap = false) {
    if (!isRecording) return;
    isRecording = false;

    // Send stop command to worklet
    workletNode.port.postMessage({ command: 'stop' });
    clearInterval(timerInterval);

    setStatus('Saved Locally / Uploading');
    btnRecord.querySelector('span').textContent = 'Record Take';
    btnRecord.disabled = true;

    // Visual tail (200ms grace period)
    await new Promise(r => setTimeout(r, 200));

    if (hitCap) {
      showAlert('Maximum take duration of 60 seconds reached. Finalized and saved take.', true);
    }

    const elapsed = Math.max(0.1, (performance.now() - recordingStartTime) / 1000);
    updateTimer(elapsed);

    // Encode WAV
    const wavBytes = encodeWAV(recordingChunks, audioContext.sampleRate);
    recordingChunks = [];

    const takeId = 'take_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);
    const sentenceId = currentPrompt ? currentPrompt.sentence_id : 'unknown';

    const takeRecord = {
      take_id: takeId,
      sentence_id: sentenceId,
      session_id: sessionId,
      wav_bytes: wavBytes,
      sample_rate: audioContext.sampleRate,
      capture_time: new Date().toISOString(),
      duration_s: elapsed,
      status: 'pending_upload',
      keep_decision: false,
    };

    activeTake = takeRecord;
    if (rerecordMode && rerecordQueue.includes(sentenceId)) {
      rerecordFreshTakeId = takeId;
    }

    // Stage immediately to IndexedDB
    try {
      await idbPutTake(takeRecord);
      await updatePendingCount();
    } catch (err) {
      console.error('CRITICAL: IndexedDB stage failed:', err);
      showAlert(`Storage quota/failure: could not stage take locally: ${err.message}. Advance blocked.`);
      btnRecord.disabled = false;
      return;
    }

    btnRecord.disabled = false;
    btnKeep.disabled = false;
    btnRetake.disabled = false;
    btnPlayback.disabled = false;
    btnPrev.disabled = false;
    btnNext.disabled = false;

    // Keep the locally staged take active while it uploads. Reloading the
    // prompt here used to race the upload, erase activeTake, and force users
    // to record every sentence twice before Keep became available.
    renderLocalPendingTake(takeRecord);
    const uploadDone = triggerUploadQueue();

    // Replace the temporary local row with authoritative server metadata once
    // upload completes, unless the user has already kept it and moved on.
    uploadDone.then(async () => {
      const queued = await idbGetTake(takeId);
      const reachedServer = !queued || queued.status !== 'pending_upload';
      if (
        reachedServer &&
        currentPrompt && currentPrompt.sentence_id === sentenceId &&
        activeTake && activeTake.take_id === takeId &&
        !isRecording
      ) {
        await loadPrompt(sentenceId);
      }
    }).catch(err => console.warn('Could not refresh uploaded take:', err));
  }

  function encodeWAV(chunks, sampleRate) {
    let totalLength = 0;
    for (let i = 0; i < chunks.length; i++) {
      totalLength += chunks[i].length;
    }

    const samples = new Float32Array(totalLength);
    let offset = 0;
    for (let i = 0; i < chunks.length; i++) {
      samples.set(chunks[i], offset);
      offset += chunks[i].length;
    }

    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');

    // "fmt " sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true);  // AudioFormat (1 for PCM)
    view.setUint16(22, 1, true);  // NumChannels (1 mono)
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // ByteRate (SampleRate * 1 * 2)
    view.setUint16(32, 2, true);  // BlockAlign (1 * 2)
    view.setUint16(34, 16, true); // BitsPerSample (16 bits)

    // "data" sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // PCM 16-bit samples
    let sampleOffset = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      const int16 = s < 0 ? s * 0x8000 : s * 0x7FFF;
      view.setInt16(sampleOffset, int16, true);
      sampleOffset += 2;
    }

    return new Uint8Array(buffer);
  }

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // -------------------------------------------------------------------------
  // Durable Upload Queue Worker
  // -------------------------------------------------------------------------

  let isUploading = false;
  let uploadPassRequested = false;

  async function triggerUploadQueue() {
    if (isUploading) {
      uploadPassRequested = true;
      return;
    }
    isUploading = true;

    try {
      do {
        uploadPassRequested = false;
        const pending = await idbGetPendingTakes();
        for (let item of pending) {
          if (item.status === 'pending_upload') {
            const success = await uploadTake(item);
            if (success) {
              item = await idbMarkUploadComplete(item.take_id);
              if (!item) continue;
            }
          }

          if (item.status === 'pending_keep') {
            const keepSuccess = await sendKeepDecision(item.take_id, item.sentence_id);
            if (keepSuccess) {
              await idbDeleteTake(item.take_id);
            }
          }
        }
      } while (uploadPassRequested);
    } catch (err) {
      console.warn('Upload queue error:', err);
    } finally {
      isUploading = false;
      await updatePendingCount();
      await updateStats();
    }
  }

  async function uploadTake(item) {
    const formData = new FormData();
    const blob = new Blob([item.wav_bytes], { type: 'audio/wav' });
    formData.append('audio', blob, `${item.take_id}.wav`);
    formData.append('take_id', item.take_id);
    formData.append('session_id', item.session_id);
    formData.append('sentence_id', item.sentence_id);
    formData.append('capture_time', item.capture_time);

    try {
      const res = await fetch('/api/takes', {
        method: 'POST',
        body: formData,
      });

      if (res.ok || res.status === 200 || res.status === 201) {
        setStatus('Saved on Server');
        return true;
      } else if (res.status === 409) {
        // Conflict
        showAlert(`Upload conflict: take ID ${item.take_id} already exists with different data.`);
        return false;
      } else {
        const data = await res.json().catch(() => ({}));
        console.warn('Upload failed:', res.status, data);
        setStatus('Upload Retry Pending', 'warning');
        return false;
      }
    } catch (err) {
      console.warn('Network error during take upload:', err);
      setStatus('Offline / Retrying', 'warning');
      return false;
    }
  }

  async function sendKeepDecision(takeId, sentenceId) {
    try {
      const res = await fetch(`/api/takes/${takeId}/keep`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence_id: sentenceId }),
      });
      return res.ok;
    } catch (err) {
      console.warn('Failed to send keep decision:', err);
      return false;
    }
  }

  // -------------------------------------------------------------------------
  // Keep, Retake, and Playback Decisions
  // -------------------------------------------------------------------------

  async function keepAndAdvance() {
    if (isRecording || !activeTake) return;

    if (activeTake.wav_bytes) {
      // A fresh local take may still be uploading. Queue an idempotent upload
      // followed by Keep so pressing Enter immediately is always safe.
      activeTake.keep_decision = true;
      activeTake.status = 'pending_upload';
      await idbPutTake(activeTake);
      triggerUploadQueue();
    } else {
      // Takes loaded from the server can be selected directly.
      const kept = await sendKeepDecision(activeTake.take_id, currentPrompt.sentence_id);
      if (!kept) {
        showAlert('Could not keep this take. Please try Enter again.', true);
        return;
      }
    }

    // Advance to next uncompleted prompt
    advanceToNext(true);
  }

  function retakeCurrent() {
    if (isRecording) return;
    // Reset active take for sentence, start new recording immediately
    startRecording();
  }

  function replayCurrentTake() {
    if (isRecording || !activeTake) return;
    if (isPlaying) {
      playbackAudio.pause();
      isPlaying = false;
      btnPlayback.querySelector('span').textContent = 'Replay';
      return;
    }

    let url;
    if (activeTake.wav_bytes) {
      const blob = new Blob([activeTake.wav_bytes], { type: 'audio/wav' });
      url = URL.createObjectURL(blob);
    } else {
      url = `/api/takes/${activeTake.take_id}/audio?type=raw`;
    }

    playbackAudio.src = url;
    playbackAudio.play();
    isPlaying = true;
    btnPlayback.querySelector('span').textContent = 'Stop Playback';
    setStatus('Playing Take');

    playbackAudio.onended = () => {
      isPlaying = false;
      btnPlayback.querySelector('span').textContent = 'Replay';
      setStatus('Ready');
    };
  }

  async function pauseSession() {
    if (isRecording) {
      await stopRecording();
    }
    if (micStream) {
      micStream.getTracks().forEach(t => t.stop());
      micStream = null;
    }
    micEnabled = false;
    btnMicEnable.textContent = 'Enable Microphone';
    btnMicEnable.disabled = false;
    btnRecord.disabled = true;
    setStatus('Session Paused');
  }

  // -------------------------------------------------------------------------
  // Navigation & Catalog Management
  // -------------------------------------------------------------------------

  async function loadCatalog() {
    try {
      const res = await fetch('/api/catalog');
      const data = await res.json();
      catalogPrompts = data.prompts || [];

      if (rerecordMode) {
        try {
          const storedQueue = JSON.parse(localStorage.getItem(RERECORD_QUEUE_KEY) || '[]');
          const validIds = new Set(catalogPrompts.map(prompt => prompt.sentence_id));
          rerecordQueue = [...new Set(storedQueue)].filter(id => validIds.has(id));
        } catch (err) {
          console.warn('Could not read re-record queue:', err);
          rerecordQueue = [];
        }

        if (rerecordQueue.length) {
          await loadPrompt(rerecordQueue[0]);
          showAlert(`Re-record queue active: ${rerecordQueue.length} sentence${rerecordQueue.length === 1 ? '' : 's'} remaining. Record a fresh take, then press Enter.`, true);
        } else {
          rerecordMode = false;
        }
      }

      if (!rerecordMode) {
        // Restore last sentence or pick first uncompleted
        const savedSentenceId = localStorage.getItem('last_sentence_id');
        if (savedSentenceId && catalogPrompts.some(p => p.sentence_id === savedSentenceId)) {
          await loadPrompt(savedSentenceId);
        } else {
          const firstUnkept = catalogPrompts.find(p => !p.is_kept) || catalogPrompts[0];
          if (firstUnkept) {
            await loadPrompt(firstUnkept.sentence_id);
          }
        }
      }

      await updateStats();
    } catch (err) {
      console.error('Failed to load catalog:', err);
      showAlert('Failed to load prompt catalog from backend.');
    }
  }

  async function loadPrompt(sentenceId) {
    try {
      const res = await fetch(`/api/prompts/${sentenceId}`);
      if (!res.ok) throw new Error('Prompt not found');
      const data = await res.json();

      currentPrompt = data;
      localStorage.setItem('last_sentence_id', sentenceId);

      currentSentenceIdEl.textContent = data.sentence_id;
      currentSplitTagEl.textContent = `Split: ${data.split}`;
      currentPromptTextEl.textContent = data.text;

      // Next preview
      if (rerecordMode) {
        const nextQueuedId = rerecordQueue.find(id => id !== data.sentence_id);
        const nextQueued = catalogPrompts.find(p => p.sentence_id === nextQueuedId);
        nextPrompt = nextQueued || null;
        nextPromptPreviewEl.textContent = nextQueued
          ? `Re-record next: (${nextQueued.sentence_id}) ${nextQueued.text}`
          : '(Last item in re-record queue)';
      } else if (data.next_sentence_id) {
        const nextP = catalogPrompts.find(p => p.sentence_id === data.next_sentence_id);
        nextPrompt = nextP;
        nextPromptPreviewEl.textContent = nextP ? `(${nextP.sentence_id}) ${nextP.text}` : '--';
      } else {
        nextPrompt = null;
        nextPromptPreviewEl.textContent = '(End of catalog)';
      }

      btnPrev.disabled = rerecordMode || !data.prev_sentence_id;
      btnNext.disabled = rerecordMode ? rerecordQueue.length <= 1 : !data.next_sentence_id;

      // Update takes drawer
      renderTakesList(data.takes || []);

      // If takes exist, set most recent as activeTake
      if (data.takes && data.takes.length > 0) {
        const sorted = [...data.takes].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
        activeTake = sorted[0];
        btnKeep.disabled = isRecording;
        btnPlayback.disabled = isRecording;
      } else {
        activeTake = null;
        btnKeep.disabled = true;
        btnPlayback.disabled = true;
      }

      // Existing kept takes may be replayed, but a queue item can advance only
      // after this re-recording session creates a genuinely fresh take.
      if (rerecordMode && rerecordQueue.includes(sentenceId)) {
        btnKeep.disabled = isRecording || !activeTake || activeTake.take_id !== rerecordFreshTakeId;
      }

      updateTimer(0);
      setStatus('Ready');
    } catch (err) {
      console.error('Failed to load prompt:', err);
      showAlert(`Could not load prompt ${sentenceId}: ${err.message}`);
    }
  }

  function renderTakesList(takes) {
    takesListEl.innerHTML = '';
    takesSummaryEl.textContent = `${takes.length} take${takes.length === 1 ? '' : 's'}`;

    if (takes.length === 0) {
      takesListEl.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem;">No takes recorded yet for this sentence.</div>';
      return;
    }

    takes.forEach((t) => {
      const row = document.createElement('div');
      row.className = 'take-row' + (t.keep_decision ? ' is-kept' : '');

      const meta = document.createElement('div');
      meta.className = 'take-meta';
      meta.innerHTML = `
        <strong>${t.take_id}</strong>
        <span>${t.duration_s}s</span>
        <span class="badge ${t.conversion_status === 'ready' ? 'badge-success' : 'badge-warning'}">
          ${t.conversion_status === 'ready' ? '16k ready' : t.conversion_status}
        </span>
        ${t.keep_decision ? '<span class="badge badge-success">KEPT</span>' : ''}
      `;

      const actions = document.createElement('div');
      actions.className = 'take-actions';

      const playBtn = document.createElement('button');
      playBtn.textContent = 'Play';
      playBtn.onclick = () => {
        playbackAudio.src = `/api/takes/${t.take_id}/audio?type=raw`;
        playbackAudio.play();
      };

      const selectBtn = document.createElement('button');
      selectBtn.textContent = t.keep_decision ? 'Kept' : 'Select as Kept';
      selectBtn.disabled = t.keep_decision === 1;
      selectBtn.onclick = async () => {
        await sendKeepDecision(t.take_id, currentPrompt.sentence_id);
        await loadPrompt(currentPrompt.sentence_id);
        await updateStats();
      };

      actions.appendChild(playBtn);
      actions.appendChild(selectBtn);
      row.appendChild(meta);
      row.appendChild(actions);
      takesListEl.appendChild(row);
    });
  }

  function renderLocalPendingTake(take) {
    if (!currentPrompt || currentPrompt.sentence_id !== take.sentence_id) return;

    if (!currentPrompt.takes || currentPrompt.takes.length === 0) {
      takesListEl.innerHTML = '';
    }

    const row = document.createElement('div');
    row.className = 'take-row';
    row.dataset.takeId = take.take_id;

    const meta = document.createElement('div');
    meta.className = 'take-meta';

    const name = document.createElement('strong');
    name.textContent = take.take_id;
    const duration = document.createElement('span');
    duration.textContent = `${take.duration_s.toFixed(2)}s`;
    const status = document.createElement('span');
    status.className = 'badge badge-warning';
    status.textContent = 'saved locally / uploading';

    meta.appendChild(name);
    meta.appendChild(duration);
    meta.appendChild(status);
    row.appendChild(meta);
    takesListEl.prepend(row);

    const serverCount = (currentPrompt.takes || []).length;
    const count = serverCount + 1;
    takesSummaryEl.textContent = `${count} take${count === 1 ? '' : 's'}`;
  }

  function advanceToNext(skipKept = true) {
    if (!currentPrompt) return;
    if (rerecordMode) {
      advanceRerecordQueue();
      return;
    }
    const currIdx = currentPrompt.order_index;
    for (let i = currIdx + 1; i < catalogPrompts.length; i++) {
      const p = catalogPrompts[i];
      if (!skipKept || !p.is_kept) {
        loadPrompt(p.sentence_id);
        return;
      }
    }
    // If none found forward, try beginning
    if (skipKept) {
      const anyUnkept = catalogPrompts.find(p => !p.is_kept);
      if (anyUnkept) {
        loadPrompt(anyUnkept.sentence_id);
        return;
      }
    }
    showAlert('All prompts in the catalog have been recorded and kept!', false);
  }

  function advanceRerecordQueue() {
    if (!currentPrompt) return;
    rerecordQueue = rerecordQueue.filter(id => id !== currentPrompt.sentence_id);
    rerecordFreshTakeId = null;

    if (rerecordQueue.length) {
      localStorage.setItem(RERECORD_QUEUE_KEY, JSON.stringify(rerecordQueue));
      loadPrompt(rerecordQueue[0]).then(() => {
        showAlert(`Re-record queue active: ${rerecordQueue.length} sentence${rerecordQueue.length === 1 ? '' : 's'} remaining.`, true);
      });
      return;
    }

    localStorage.removeItem(RERECORD_QUEUE_KEY);
    rerecordMode = false;
    btnKeep.disabled = true;
    history.replaceState({}, '', '/');
    showAlert('Re-record queue complete. Every replacement take was kept.', false);
  }

  function advanceToPrev() {
    if (!currentPrompt || !currentPrompt.prev_sentence_id) return;
    loadPrompt(currentPrompt.prev_sentence_id);
  }

  btnPrev.addEventListener('click', advanceToPrev);
  btnNext.addEventListener('click', () => advanceToNext(chkSkipKept.checked));

  btnJump.addEventListener('click', () => {
    const target = inputJump.value.trim();
    if (target && catalogPrompts.some(p => p.sentence_id === target)) {
      loadPrompt(target);
      inputJump.value = '';
    } else {
      showAlert(`Sentence ID '${target}' not found in catalog.`);
    }
  });

  inputJump.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      btnJump.click();
    }
  });

  // -------------------------------------------------------------------------
  // Writer Lease & Multi-Tab Synchronization
  // -------------------------------------------------------------------------

  async function acquireLease() {
    try {
      const res = await fetch('/api/lease/acquire', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client_id: clientId, ttl_seconds: 15 }),
      });

      if (res.ok) {
        leaseAcquired = true;
        leaseBadge.textContent = 'Writer Lease: Active';
        leaseBadge.className = 'badge badge-success';
        if (micEnabled) btnRecord.disabled = false;
        startLeaseHeartbeat();
      } else if (res.status === 409) {
        leaseAcquired = false;
        leaseBadge.textContent = 'Writer Lease: Conflict';
        leaseBadge.className = 'badge badge-danger';
        btnRecord.disabled = true;
        showAlert('Another browser tab is actively controlling this dataset. Recording is disabled here.', true);
      }
    } catch (err) {
      console.warn('Lease acquisition failed:', err);
    }
  }

  function startLeaseHeartbeat() {
    clearInterval(leaseInterval);
    leaseInterval = setInterval(async () => {
      try {
        const res = await fetch('/api/lease/heartbeat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ client_id: clientId, ttl_seconds: 15 }),
        });
        if (!res.ok) {
          leaseAcquired = false;
          leaseBadge.textContent = 'Writer Lease: Lost';
          leaseBadge.className = 'badge badge-danger';
          btnRecord.disabled = true;
          showAlert('Writer lease lost. Another tab has taken control.', true);
        }
      } catch (err) {
        console.warn('Heartbeat error:', err);
      }
    }, 5000);
  }

  window.addEventListener('beforeunload', () => {
    if (leaseAcquired) {
      navigator.sendBeacon('/api/lease/release', JSON.stringify({ client_id: clientId }));
    }
  });

  // -------------------------------------------------------------------------
  // Stats & Dataset Metadata
  // -------------------------------------------------------------------------

  async function updateStats() {
    try {
      const res = await fetch('/api/dataset');
      if (!res.ok) return;
      const data = await res.json();
      const stats = data.stats || {};
      const meta = data.metadata || {};

      speakerBadge.textContent = `Speaker: ${meta.speaker || 'PERSONAL'}`;
      splitBadge.textContent = `Split: ${meta.split_provenance ? (meta.split_provenance.includes('/') ? meta.split_provenance.split('/').pop() : meta.split_provenance) : 'unassigned'}`;
      keptCountEl.textContent = `${stats.kept_prompts || 0} / ${stats.total_prompts || 1132}`;
      serverTakesCountEl.textContent = (stats.total_takes || 0).toString();
      reviewBadge.textContent = `Transcript Verification: ${stats.review_status || 'unreviewed'}`;
    } catch (err) {
      console.warn('Failed to update stats:', err);
    }
  }

  // -------------------------------------------------------------------------
  // Keyboard Shortcuts & Event Handlers
  // -------------------------------------------------------------------------

  btnRecord.addEventListener('click', () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  btnKeep.addEventListener('click', keepAndAdvance);
  btnRetake.addEventListener('click', retakeCurrent);
  btnPlayback.addEventListener('click', replayCurrentTake);
  btnPauseSession.addEventListener('click', pauseSession);

  window.addEventListener('keydown', (e) => {
    // Ignore key repeat events
    if (e.repeat) return;

    // Ignore when focus is inside text input or select
    const tag = e.target.tagName.toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea') {
      return;
    }

    if (e.code === 'Space') {
      e.preventDefault(); // Suppress page scroll
      if (isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    } else if (e.code === 'Enter') {
      e.preventDefault();
      keepAndAdvance();
    } else if (e.key === 'r' || e.key === 'R') {
      e.preventDefault();
      retakeCurrent();
    } else if (e.key === 'p' || e.key === 'P') {
      e.preventDefault();
      replayCurrentTake();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      pauseSession();
    }
  });

  // -------------------------------------------------------------------------
  // Application Bootstrap
  // -------------------------------------------------------------------------

  async function init() {
    await acquireLease();
    await loadCatalog();
    await updatePendingCount();
    // Run upload queue for any leftovers in IndexedDB
    triggerUploadQueue();
  }

  window.addEventListener('DOMContentLoaded', init);
})();
