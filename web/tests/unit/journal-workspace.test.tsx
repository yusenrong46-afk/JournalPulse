import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const navigation = vi.hoisted(() => ({ query: "", listeners: new Set<() => void>(), replace: vi.fn() }));

vi.mock("next/navigation", async () => {
  const { useMemo, useSyncExternalStore } = await import("react");
  const router = { replace: navigation.replace };
  return {
    useRouter: () => router,
    useSearchParams: () => {
      const query = useSyncExternalStore((listener) => {
        navigation.listeners.add(listener);
        return () => { navigation.listeners.delete(listener); };
      }, () => navigation.query);
      return useMemo(() => new URLSearchParams(query), [query]);
    },
  };
});

vi.mock("next/link", () => ({
  default: ({ href, children, scroll: _scroll, ...props }: {
    href: string; children: ReactNode; scroll?: boolean;
  }) => {
    // Next.js handles scroll itself; it is not an HTML attribute.
    void _scroll;
    return createElement("a", { href, ...props }, children);
  },
}));

vi.mock("@/lib/journal", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/journal")>(),
  listJournalEntries: vi.fn(), readJournalEntry: vi.fn(), saveJournalEntry: vi.fn(),
  reflectJournalEntry: vi.fn(), deleteJournalEntry: vi.fn(),
}));

import JournalPage from "@/app/journal/page";
import { ApiError } from "@/lib/api";
import { listJournalEntries, readJournalEntry, reflectJournalEntry, saveJournalEntry } from "@/lib/journal";
import type { JournalEntry, JournalReflectionResult } from "@/lib/journal-types";
import { clearTabSession } from "@/lib/tab-session";
import { ACCOUNT_DATA_CHANGED_EVENT } from "@/lib/account-data";

const entry: JournalEntry = {
  id: "10000000-0000-4000-8000-000000000001", user_id: "owner",
  created_at: "2026-10-04T12:00:00Z", text: "My exact writing.\nThe line break matters.",
};
const other: JournalEntry = { ...entry, id: "10000000-0000-4000-8000-000000000002", text: "Another moment." };
const reflection: JournalReflectionResult = {
  entry_id: entry.id, reply: "OLD_TEMPORARY_REFLECTION", generated_text_retained: false,
  safety: { mode: "normal", reasons: [], locale: "CA", exploration_allowed: true, resource_ids: [] },
  model_run: { model: "test-only", provider: "test-double", latency_ms: 0, schema_valid: true, used_fallback: false },
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

let root: Root;
let container: HTMLDivElement;

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
  window.sessionStorage.clear(); clearTabSession();
  window.dispatchEvent(new Event(ACCOUNT_DATA_CHANGED_EVENT));
  navigation.query = `entry=${entry.id}`;
  navigation.replace.mockImplementation((url: string) => {
    navigation.query = url.split("?")[1] ?? "";
    navigation.listeners.forEach((listener) => listener());
  });
  vi.mocked(listJournalEntries).mockReset().mockResolvedValue({ items: [entry, other], limit: 50, offset: 0 });
  vi.mocked(readJournalEntry).mockReset().mockImplementation(async (id) => id === entry.id ? entry : other);
  vi.mocked(reflectJournalEntry).mockReset().mockResolvedValue(reflection);
  vi.mocked(saveJournalEntry).mockReset().mockResolvedValue(entry);
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
  navigation.listeners.clear();
  vi.restoreAllMocks();
});

async function mount() {
  await act(async () => { root.render(createElement(JournalPage)); });
}

function button(label: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((item) => item.textContent?.trim() === label);
  expect(found, `Button '${label}' should be visible`).toBeTruthy();
  return found!;
}

async function click(label: string) {
  await act(async () => { button(label).click(); });
}

async function consent() {
  await act(async () => { container.querySelector<HTMLInputElement>("input[type=checkbox]")!.click(); });
}

async function write(text: string) {
  const input = container.querySelector("textarea")!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(input, text);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

describe("journal save, reflection and discussion", () => {
  test("explains saved writing and temporary reflection before consent, with a clear chat handoff", async () => {
    await mount();
    expect(container.textContent).toContain("Saved writing");
    expect(container.textContent).toContain("This reply disappears when you leave this entry or reload");
    expect(container.textContent).toContain("Chat uses your saved entry");
    expect(container.textContent).toContain("the temporary reflection is not carried into the conversation");
    expect(button("Reflect on this entry").disabled).toBe(true);
    expect(container.querySelector(`a[href='/talk?entry=${entry.id}']`)?.textContent).toBe("Discuss with Luna");
    expect(vi.mocked(reflectJournalEntry)).not.toHaveBeenCalled();
  });

  test("failed save preserves exact draft and its retry identity until a confirmed save", async () => {
    navigation.query = "";
    vi.mocked(saveJournalEntry).mockRejectedValueOnce(new ApiError("Network unavailable", 503));
    await mount();
    await write(entry.text);
    await click("Save entry");
    expect(container.querySelector("textarea")!.value).toBe(entry.text);
    expect(container.textContent).toContain("Unsaved writing");
    expect(container.textContent).not.toContain("Entry saved.");
    await click("Save entry");
    const calls = vi.mocked(saveJournalEntry).mock.calls;
    expect(calls).toHaveLength(2);
    expect(calls[0]).toEqual(calls[1]);
    expect(calls[0][0]).toBe(entry.text);
    expect(container.querySelector("textarea")).toBeNull(); // The confirmed save opens its immutable reading view.
    expect(navigation.query).toBe(`entry=${entry.id}`);
    expect(container.querySelector(".journal-entry-text")!.textContent).toBe(entry.text);
  });

  test("reflection failure keeps saved writing and the discussion path available", async () => {
    vi.mocked(reflectJournalEntry).mockRejectedValueOnce(new ApiError("Private AI is unavailable", 503));
    await mount();
    await consent();
    await click("Reflect on this entry");
    expect(container.textContent).toContain("Private AI is unavailable");
    expect(container.textContent).toContain("Saved writing");
    expect(container.querySelector(".journal-entry-text")!.textContent).toBe(entry.text);
    expect(container.querySelector(`a[href='/talk?entry=${entry.id}']`)).toBeTruthy();
    expect(container.querySelector(".journal-reflection")).toBeNull();
  });

  test("a save that finishes after leaving the journal cannot navigate back", async () => {
    navigation.query = "";
    const delayed = deferred<JournalEntry>();
    vi.mocked(saveJournalEntry).mockReturnValueOnce(delayed.promise);
    await mount();
    await write(entry.text);
    await click("Save entry");
    await act(async () => { root.render(createElement("p", null, "Another page")); });
    navigation.replace.mockClear();
    await act(async () => { delayed.resolve(entry); });
    expect(navigation.replace).not.toHaveBeenCalled();
    expect(container.textContent).toBe("Another page");
  });

  test("unsaved writing survives leaving and returning without a save", async () => {
    navigation.query = "";
    await mount(); await write(entry.text);
    await act(async () => root.render(null));
    await mount();
    expect(container.querySelector("textarea")!.value).toBe(entry.text);
    expect(vi.mocked(saveJournalEntry)).not.toHaveBeenCalled();
  });

  test("save completion reconciles a returned list and preserves newer writing", async () => {
    navigation.query = "";
    const delayed = deferred<JournalEntry>();
    vi.mocked(saveJournalEntry).mockReturnValueOnce(delayed.promise);
    await mount(); await write(entry.text); await click("Save entry");
    await act(async () => root.render(null)); await mount();
    await write("A newer unsaved thought.");
    vi.mocked(listJournalEntries).mockResolvedValue({ items: [entry, other], limit: 50, offset: 0 });
    await act(async () => delayed.resolve(entry));
    expect(container.querySelector("textarea")!.value).toBe("A newer unsaved thought.");
    expect(container.textContent).toContain("Entry saved.");
    expect(container.querySelector(`a[href='/journal?entry=${entry.id}']`)).toBeTruthy();
    expect(navigation.query).toBe("");
  });

  test("provider decline shows its explanation without hiding writing or inventing a reply", async () => {
    const message = "Luna's AI provider declined this reflection. Your entry is still saved. "
      + "You can keep journaling without an AI reply.";
    vi.mocked(reflectJournalEntry).mockRejectedValueOnce(new ApiError(message, 422));
    await mount();
    await consent();
    await click("Reflect on this entry");
    expect(container.querySelector("[role=alert]")?.textContent).toBe(message);
    expect(container.querySelector(".journal-entry-text")!.textContent).toBe(entry.text);
    expect(container.querySelector(".journal-reflection")).toBeNull();
    expect(vi.mocked(reflectJournalEntry)).toHaveBeenCalledTimes(1);
  });

  test("changing entries removes temporary reflection and requires fresh consent", async () => {
    await mount();
    await consent();
    await click("Reflect on this entry");
    expect(container.textContent).toContain(reflection.reply);
    await act(async () => { navigation.replace(`/journal?entry=${other.id}`); });
    expect(container.querySelector(".journal-entry-text")!.textContent).toBe(other.text);
    expect(container.textContent).not.toContain(reflection.reply);
    expect(container.querySelector<HTMLInputElement>("input[type=checkbox]")!.checked).toBe(false);
    expect(button("Reflect on this entry").disabled).toBe(true);
  });

  test("a delayed reflection for the previous entry cannot appear in the newly selected entry", async () => {
    const delayed = deferred<JournalReflectionResult>();
    vi.mocked(reflectJournalEntry).mockReturnValueOnce(delayed.promise);
    await mount();
    await consent();
    await click("Reflect on this entry");
    const oldSignal = vi.mocked(reflectJournalEntry).mock.calls[0][3]!;
    await act(async () => { navigation.replace(`/journal?entry=${other.id}`); });
    expect(oldSignal.aborted).toBe(true);
    await act(async () => { delayed.resolve(reflection); });
    expect(container.querySelector(".journal-entry-text")!.textContent).toBe(other.text);
    expect(container.textContent).not.toContain(reflection.reply);
    expect(button("Reflect on this entry").disabled).toBe(true);
  });

  test("remounting a saved entry restores writing but no reflection or consent", async () => {
    await mount();
    await consent();
    await click("Reflect on this entry");
    expect(container.textContent).toContain(reflection.reply);
    await act(async () => { root.unmount(); });
    root = createRoot(container);
    await mount();
    expect(container.querySelector(".journal-entry-text")!.textContent).toBe(entry.text);
    expect(container.textContent).not.toContain(reflection.reply);
    expect(container.querySelector<HTMLInputElement>("input[type=checkbox]")!.checked).toBe(false);
  });
});

test("entry dates say how long ago they were written", async () => {
  const { entryDateLabel } = await import("@/lib/journal");
  const now = new Date(2026, 9, 5, 9);
  expect(entryDateLabel(new Date(2026, 9, 5, 1).toISOString(), now).age).toBe("written today");
  expect(entryDateLabel(new Date(2026, 9, 4, 23).toISOString(), now).age).toBe("written yesterday");
  expect(entryDateLabel(new Date(2026, 8, 28, 12).toISOString(), now).age).toBe("written 7 days ago");
});
