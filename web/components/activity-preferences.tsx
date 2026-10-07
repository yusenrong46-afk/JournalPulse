"use client";

import { useEffect, useRef, useState, type FormEvent, type RefObject } from "react";
import type { ActivityConstraints } from "@/lib/types";

const DEFAULT_LIMITS: ActivityConstraints = {
  time_minutes: null, no_audio: false, no_video: false, seated: false, avoid_breath_focus: false,
};
const CHOICES = [
  ["no_audio", "No audio"], ["no_video", "No video"],
  ["seated", "Stay seated"], ["avoid_breath_focus", "Avoid breath-focused activities"],
] as const;

export function activityPreferencesMessage(limits: ActivityConstraints): string {
  const parts = [
    ...(limits.time_minutes ? [`up to ${limits.time_minutes} minute${limits.time_minutes === 1 ? "" : "s"}`] : []),
    ...CHOICES.filter(([key]) => limits[key]).map(([, label]) => label.toLowerCase()),
  ];
  return parts.length ? `For new activity ideas: ${parts.join(", ")}.` : "Please clear my activity limits for new suggestions.";
}

export function ActivityPreferences({ current, busy, opener, onClose, onApply }: {
  current?: ActivityConstraints; busy: boolean; opener: RefObject<HTMLButtonElement | null>;
  onClose(): void; onApply(limits: ActivityConstraints): void;
}) {
  const [limits, setLimits] = useState<ActivityConstraints>(() => ({ ...DEFAULT_LIMITS, ...current }));
  const dialog = useRef<HTMLFormElement>(null);
  const close = useRef(onClose);
  useEffect(() => { close.current = onClose; }, [onClose]);
  useEffect(() => {
    const element = dialog.current;
    const trigger = opener.current;
    if (!element) return;
    const controls = () => Array.from(element.querySelectorAll<HTMLElement>("button:not([disabled]), input:not([disabled]), select:not([disabled])"));
    controls()[0]?.focus();
    function keydown(event: KeyboardEvent) {
      if (event.key === "Escape") { event.preventDefault(); close.current(); }
      if (event.key !== "Tab") return;
      const items = controls();
      const first = items[0], last = items.at(-1);
      if (!first || !last) return;
      if (!element!.contains(document.activeElement) || (event.shiftKey && document.activeElement === first)
        || (!event.shiftKey && document.activeElement === last)) {
        event.preventDefault(); (event.shiftKey ? last : first).focus();
      }
    }
    document.addEventListener("keydown", keydown);
    return () => { document.removeEventListener("keydown", keydown); if (trigger?.isConnected) trigger.focus(); };
  }, [opener]);
  function submit(event: FormEvent) { event.preventDefault(); if (!busy) onApply(limits); }
  return <div className="sheet-backdrop" onClick={onClose}>
    <form ref={dialog} className="sheet stack" role="dialog" aria-modal="true"
      aria-labelledby="activity-preferences-title" aria-label="Activity preferences"
      onSubmit={submit} onClick={(event) => event.stopPropagation()}>
      <h2 id="activity-preferences-title">What fits today?</h2>
      <p className="muted small">Choose limits for new activity ideas in this chat. You can change or clear them here anytime.</p>
      <fieldset disabled={busy} className="stack activity-preferences-fields">
        <label className="text-field">Time I have
          <select value={limits.time_minutes ?? ""} onChange={(event) => setLimits({ ...limits, time_minutes: event.target.value ? Number(event.target.value) : null })}>
            <option value="">No time limit</option>
            {Array.from({ length: 20 }, (_, i) => i + 1).map((minutes) => <option key={minutes} value={minutes}>{minutes} minute{minutes === 1 ? "" : "s"}</option>)}
          </select>
        </label>
        {CHOICES.map(([key, label]) => <label className="check-row" key={key}>
          <input type="checkbox" checked={limits[key]} onChange={(event) => setLimits({ ...limits, [key]: event.target.checked })} />
          <span>{label}</span>
        </label>)}
        <button className="btn btn-primary" type="submit">Use these preferences</button>
      </fieldset>
      <button className="btn btn-ghost" type="button" onClick={onClose}>Cancel</button>
    </form>
  </div>;
}
