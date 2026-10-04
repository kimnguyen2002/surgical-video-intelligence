/**
 * A one-hop channel from the vision worker to the overlay.
 *
 * Detections arrive as fast as the models can produce them — up to video frame
 * rate. Pushing each one through React state at the top of the app would
 * re-render the case library, the ground-truth ribbon and the chat transcript
 * dozens of times a second to move two rectangles.
 *
 * So the overlay subscribes here and re-renders alone, and the rest of the
 * interface takes a throttled copy. The panels do not need 60 Hz; the boxes
 * drawn over a moving instrument do.
 */

import { LocalVisionResult } from './localVision';

type Listener = (result: LocalVisionResult | null) => void;

const listeners = new Set<Listener>();
let latest: LocalVisionResult | null = null;

export const liveVisionBus = {
  subscribe(listener: Listener): () => void {
    listeners.add(listener);
    listener(latest);
    return () => {
      listeners.delete(listener);
    };
  },

  publish(result: LocalVisionResult | null) {
    latest = result;
    for (const listener of listeners) listener(result);
  },

  get current() {
    return latest;
  },
};
