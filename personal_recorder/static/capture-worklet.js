/**
 * AudioWorklet processor for streaming raw PCM samples without lossy compression.
 */
class CaptureWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.isRecording = false;
    this.port.onmessage = (event) => {
      if (event.data.command === 'start') {
        this.isRecording = true;
      } else if (event.data.command === 'stop') {
        this.isRecording = false;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) {
      return true;
    }

    const channelData = input[0];
    if (channelData && channelData.length > 0) {
      // Calculate RMS for level metering
      let sumSquares = 0;
      for (let i = 0; i < channelData.length; i++) {
        sumSquares += channelData[i] * channelData[i];
      }
      const rms = Math.sqrt(sumSquares / channelData.length);
      this.port.postMessage({ type: 'meter', rms: rms });

      // If actively recording, transfer copy of PCM samples
      if (this.isRecording) {
        const copy = new Float32Array(channelData);
        this.port.postMessage({ type: 'pcm_chunk', samples: copy }, [copy.buffer]);
      }
    }

    return true;
  }
}

registerProcessor('capture-worklet-processor', CaptureWorkletProcessor);
