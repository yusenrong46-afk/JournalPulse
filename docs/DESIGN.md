# Design

JournalPulse should feel like a deep breath: warm, quiet, and never clinical. Someone opening it on a
hard day should know what to do within a second.

## Principles

- **One thing at a time.** Each screen asks one question. No step counters or research vocabulary.
- **Taps before typing.** Mood faces, feeling buttons, and goal buttons carry most answers; typing is
  always available.
- **The person decides.** Luna suggests; people confirm or change every feeling and choice.
- **Calm over clever.** Slow motion, soft shapes, and generous space. Playfulness never appears in
  support mode.

## Colour

| Token | Value | Use |
|---|---|---|
| `--sand` | `#fbf5ec` | Page background |
| `--plum` | `#2b2540` | Text |
| `--plum-soft` | `#5c5470` | Secondary text |
| `--sun` | `#ff9a5a` | Main buttons, selection |
| `--lav` / `--lav-soft` | `#b9a8f0` / `#efeafd` | Luna's bubbles, chips, links |
| `--sage` / `--sage-soft` | `#9fc8a8` / `#e4f1e6` | Growth, success, the timer |
| `--peach` | `#ffb4a2` | Luna's cheeks, accents |

The sky gradient at the top of Home, Chat, and onboarding changes with the reader's local time
(`data-time` on `<html>`: morning, afternoon, evening, night). All text meets WCAG AA contrast; the
browser tests run axe on every page.

## Type and shape

- Headings and buttons: **Nunito** 700–800. Body: **Inter**. Both load through `next/font`.
- Corners: 14–32 px. Buttons are pills at least 52 px tall; tap targets are at least 44 px.
- Cards are white with a soft shadow; tinted cards (`lav`, `sage`, `sun`) have no shadow.

## Motion

Luna floats on a 4.5-second cycle and blinks every few seconds. Messages rise in gently. Nothing flashes
or moves quickly. With **reduce motion** turned on, every animation stops and decorative particles are
hidden.

## Luna

Luna is a small moon-mochi with a sprout on its head, drawn as an SVG in `web/components/luna.tsx`. Its
mood follows the app:

| Mood | Shown when |
|---|---|
| `idle` | Default |
| `listening` | The person is typing (sprout perks up) |
| `thinking` | Waiting for a reply ("hmm…" bubble and orbiting stars) |
| `answering` | A reply just arrived |
| `proud` | An action was chosen or a check-in saved (falling petals) |
| `oops` | An error or no connection |
| `sleepy` | Home at night, and while the session loads |
| `checkin` | A check-in is due |
| `support` | Support mode only: calm eyes and a lantern, no jokes or particles |

Every Luna has an accessible label ("Luna is thinking") unless nearby text already says the same thing.
The concept sheet is in the pull request that introduced the redesign.

## Journey garden

Each saved check-in is a pot in the garden (`web/components/plant.tsx`):

| Plant | Meaning |
|---|---|
| Seed | Check-in still waiting |
| Sprout | Skipped, or it didn't help much |
| Leafy | It helped somewhat |
| Flower | It helped, or helped a lot |

## Voice

Plain, warm, and short. Say what the person can do next. Never diagnose or promise results.

| Instead of | Say |
|---|---|
| "The system's read is a proposal, not a verdict." | "Which of these feel true right now?" |
| "Baseline pick · eligible for policy evaluation" | "Luna's pick" |
| "Record the outcome" | "How much did it help?" |
| "Enter your private field journal." | "Welcome to JournalPulse" |
| "Delete all journal data" | "Delete everything" |

Research and privacy detail lives on the Me page under "How Luna works", not in the main flow.
