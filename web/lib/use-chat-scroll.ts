"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/** Distance from the bottom that still counts as "reading the latest message". */
const NEAR_BOTTOM_PX = 96;

/**
 * Keep the log pinned to the newest content only while the person is already at the bottom.
 * Someone reading earlier messages is not pulled down; they get a "new messages" prompt
 * instead. Content growth inside activity panels (search results, the check-in form) counts
 * too, which a dependency list of chat state could not see.
 */
export function useStickToBottom() {
  const log = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const [unseen, setUnseen] = useState(false);

  const scrollToLatest = useCallback((smooth = false) => {
    const element = log.current;
    if (!element) return;
    pinned.current = true;
    setUnseen(false);
    if (smooth && typeof element.scrollTo === "function") {
      element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
    } else {
      element.scrollTop = element.scrollHeight;
    }
  }, []);

  useEffect(() => {
    const element = log.current;
    const inner = content.current;
    if (!element || !inner) return;
    const onScroll = () => {
      const near = element.scrollHeight - element.scrollTop - element.clientHeight <= NEAR_BOTTOM_PX;
      pinned.current = near;
      if (near) setUnseen(false);
    };
    const onGrow = () => {
      if (pinned.current) element.scrollTop = element.scrollHeight;
      else setUnseen(true);
    };
    element.addEventListener("scroll", onScroll, { passive: true });
    // jsdom and very old browsers lack ResizeObserver; they keep the initial pinned scroll.
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(onGrow);
    observer?.observe(inner);
    onGrow();
    return () => {
      element.removeEventListener("scroll", onScroll);
      observer?.disconnect();
    };
  }, []);

  return { logRef: log, contentRef: content, unseen, scrollToLatest };
}

/**
 * iOS Safari keeps the layout viewport when the keyboard opens, so a 100dvh chat would hide its
 * composer under the keyboard. Size the chat to the visual viewport instead. Browsers that honour
 * `interactive-widget=resizes-content` already resize, and this then simply matches.
 */
export function useVisualViewportHeight() {
  const target = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const viewport = window.visualViewport;
    const element = target.current;
    if (!viewport || !element) return;
    const update = () => {
      element.style.setProperty("--chat-height", `${Math.round(viewport.height)}px`);
      // Undo the page scroll iOS applies to reveal a focused field; the chat already fits.
      if (window.scrollY !== 0) window.scrollTo(0, 0);
    };
    update();
    viewport.addEventListener("resize", update);
    return () => {
      viewport.removeEventListener("resize", update);
      element.style.removeProperty("--chat-height");
    };
  }, []);
  return target;
}
