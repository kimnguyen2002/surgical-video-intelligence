/**
 * Media Capture & Frame Extraction Services
 */

export class MediaService {
  /**
   * Captures a base64 JPEG from a video element or canvas element
   */
  public static captureFrameFromElement(
    sourceElement: HTMLVideoElement | HTMLCanvasElement
  ): string {
    const canvas = document.createElement('canvas');
    let width = 640;
    let height = 360;

    if (sourceElement instanceof HTMLVideoElement) {
      width = sourceElement.videoWidth || 640;
      height = sourceElement.videoHeight || 360;
    } else if (sourceElement instanceof HTMLCanvasElement) {
      width = sourceElement.width || 640;
      height = sourceElement.height || 360;
    }

    // Scale to reasonable resolution (max 960px width for fast latency)
    const scale = Math.min(1, 960 / Math.max(width, 1));
    canvas.width = Math.floor(width * scale);
    canvas.height = Math.floor(height * scale);

    const ctx = canvas.getContext('2d');
    if (!ctx) return '';

    ctx.drawImage(sourceElement, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
  }

  /**
   * Request live webcam / laparoscopic capture device stream
   */
  public static async getCameraStream(): Promise<MediaStream> {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera device access is not supported by your browser.');
    }

    return await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user',
      },
      audio: false,
    });
  }

  /**
   * Format seconds to HH:MM:SS or MM:SS
   */
  public static formatTime(seconds: number): string {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hrs > 0) {
      return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
}
