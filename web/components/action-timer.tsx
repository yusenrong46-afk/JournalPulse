"use client";

import { useEffect, useRef, useState } from "react";

const RADIUS = 64;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export function ActionTimer({ minutes, onDone }: { minutes: number; onDone?: () => void }) {
  const total = Math.max(1, Math.round(minutes)) * 60;
  const [left, setLeft] = useState(total);
  const [running, setRunning] = useState(false);
  const remainingMillis = useRef(total * 1000);
  const deadline = useRef<number | null>(null);
  const completed = useRef(false);
  const onDoneRef = useRef(onDone);
  useEffect(() => { onDoneRef.current = onDone; }, [onDone]);

  useEffect(() => {
    remainingMillis.current = total * 1000;
    deadline.current = null;
    completed.current = false;
    queueMicrotask(() => { setLeft(total); setRunning(false); });
  }, [total]);

  useEffect(() => {
    if (!running) return;
    function repaint() {
      if (deadline.current === null) return;
      remainingMillis.current = Math.max(0, deadline.current - performance.now());
      const next = Math.ceil(remainingMillis.current / 1000);
      setLeft(next);
      if (next === 0 && !completed.current) {
        completed.current = true;
        deadline.current = null;
        setRunning(false);
        // A side effect never runs inside a React state updater, including StrictMode replay.
        onDoneRef.current?.();
      }
    }
    const interval = window.setInterval(repaint, 250);
    const wake = () => { if (document.visibilityState !== "hidden") repaint(); };
    window.addEventListener("focus", wake);
    document.addEventListener("visibilitychange", wake);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("focus", wake);
      document.removeEventListener("visibilitychange", wake);
    };
  }, [running]);

  function toggle() {
    if (running) {
      remainingMillis.current = Math.max(0, (deadline.current ?? performance.now()) - performance.now());
      deadline.current = null;
      setLeft(Math.ceil(remainingMillis.current / 1000));
      setRunning(false);
    } else {
      deadline.current = performance.now() + remainingMillis.current;
      setRunning(true);
    }
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
            {running ? "Pause" : left === total ? `Start a ${Math.round(total / 60)}-minute timer` : "Keep going"}
          </button>
        )}
        {left !== total && (
          <button className="btn btn-ghost" type="button" onClick={() => {
            remainingMillis.current = total * 1000;
            deadline.current = null;
            completed.current = false;
            setRunning(false); setLeft(total);
          }}>
            Reset
          </button>
        )}
      </div>
    </div>
  );
}
