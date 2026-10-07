"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useTabValue } from "@/lib/tab-session";

const RADIUS = 64;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

type TimerState = { total: number; remaining: number; deadline: number | null };

export function ActionTimer({ minutes, onDone, sessionKey }: { minutes: number; onDone?: () => void; sessionKey?: string }) {
  const total = Math.max(1, Math.round(minutes)) * 60;
  const [stored, store] = useTabValue(`timer:${sessionKey ?? "unpersisted"}`);
  const [local, setLocal] = useState<TimerState | null>(null);
  const initial = useMemo(() => ({ total, remaining: total * 1000, deadline: null }), [total]);
  const state = useMemo(() => {
    if (!sessionKey) return local?.total === total ? local : initial;
    try {
      const value = JSON.parse(stored) as TimerState;
      if (value.total === total && Number.isFinite(value.remaining) && value.remaining >= 0 && value.remaining <= total * 1000
        && (value.deadline === null || (Number.isFinite(value.deadline) && value.deadline > 0))) return value;
    } catch { /* An absent/invalid record starts a paused timer. */ }
    return initial;
  }, [stored, sessionKey, local, initial, total]);
  const [left, setLeft] = useState(total);
  const completed = useRef<number | null>(null);
  const onDoneRef = useRef(onDone);
  useEffect(() => { onDoneRef.current = onDone; }, [onDone]);
  const running = state.deadline !== null && left > 0;
  const now = () => sessionKey ? Date.now() : performance.now();

  function save(next: TimerState) {
    if (sessionKey) store(JSON.stringify(next));
    else setLocal(next);
    setLeft(Math.ceil((next.deadline === null ? next.remaining : Math.max(0, next.deadline - now())) / 1000));
  }

  useEffect(() => {
    const repaint = () => {
      const next = Math.ceil((state.deadline === null ? state.remaining : Math.max(0, state.deadline - (sessionKey ? Date.now() : performance.now()))) / 1000);
      setLeft(next);
      if (next === 0 && state.deadline !== null && completed.current !== state.deadline) {
        completed.current = state.deadline;
        onDoneRef.current?.();
      }
    };
    queueMicrotask(repaint);
    const interval = window.setInterval(repaint, 250);
    const wake = () => { if (document.visibilityState !== "hidden") repaint(); };
    window.addEventListener("focus", wake);
    document.addEventListener("visibilitychange", wake);
    return () => { window.clearInterval(interval); window.removeEventListener("focus", wake); document.removeEventListener("visibilitychange", wake); };
  }, [state, sessionKey]);

  function toggle() {
    if (running) save({ total, remaining: Math.max(0, state.deadline! - now()), deadline: null });
    else save({ total, remaining: state.remaining, deadline: now() + state.remaining });
  }

  const progress = 1 - left / total;
  const label = `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;

  return (
    <div className="timer">
      <svg viewBox="0 0 150 150" aria-hidden="true">
        <circle cx="75" cy="75" r={RADIUS} fill="none" stroke="var(--sand-deep)" strokeWidth="12" />
        <circle
          cx="75"
          cy="75"
          r={RADIUS}
          fill="none"
          stroke="var(--sage)"
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={CIRCUMFERENCE * (1 - progress)}
          style={{ transition: "stroke-dashoffset 1s linear" }}
        />
      </svg>
      <div className="timer-text" role="timer" aria-live="off">{label}</div>
      <div className="row">
        {left === 0 ? (
          <span className="tag sage" role="status">Time’s up. You can check in when you’re ready.</span>
        ) : (
          <button className="btn btn-soft" type="button" onClick={toggle}>
            {running ? "Pause" : state.remaining === total * 1000 ? `Start a ${Math.round(total / 60)}-minute timer` : "Keep going"}
          </button>
        )}
        {(left !== total || state.remaining !== total * 1000 || state.deadline !== null) && (
          <button className="btn btn-ghost" type="button" onClick={() => {
            completed.current = null;
            save(initial);
          }}>
            Reset
          </button>
        )}
      </div>
    </div>
  );
}
