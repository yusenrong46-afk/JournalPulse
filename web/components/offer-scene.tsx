import { Luna } from "./luna";

/** The invitation's small illustrated scene: a window, warm lamp light and Luna resting. Decorative. */
export function OfferScene() {
  return (
    <div className="offer-scene" aria-hidden="true">
      <span className="offer-scene-window" />
      <span className="offer-scene-glow" />
      <span className="offer-scene-cushion" />
      <span className="offer-scene-luna"><Luna mood="resting" size={60} decorative /></span>
      <span className="offer-scene-tag">Optional</span>
    </div>
  );
}
