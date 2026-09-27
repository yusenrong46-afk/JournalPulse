"use client";

import { useEffect, useState } from "react";

const RADIUS = 64;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export function ActionTimer({ minutes, onDone }: { minutes: number; onDone?: () => void }) {
  const total = Math.max(1, Math.round(minutes)) * 60;
  const [left, setLeft] = useState(total);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (!running) return;
    const interval = window.setInterval(() => {
      setLeft((value) => {
        if (value <= 1) {
          window.clearInterval(interval);
          setRunning(false);
          onDone?.();
          return 0;
        }
        return value - 1;
      });
    }, 1000);
    return () => window.clearInterval(interval);
  }, [onDone, running]);

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
          <span className="tag sage">Time’s up. Nice work.</span>
        ) : (
          <button className="btn btn-soft" type="button" onClick={() => setRunning((value) => !value)}>
            {running ? "Pause" : left === total ? `Start a ${Math.round(total / 60)}-minute timer` : "Keep going"}
          </button>
        )}
        {left !== total && (
          <button className="btn btn-ghost" type="button" onClick={() => { setRunning(false); setLeft(total); }}>
            Reset
          </button>
        )}
      </div>
    </div>
  );
}
