import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const navigation = vi.hoisted(() => ({
  query: "",
  listeners: new Set<() => void>(),
  replace: vi.fn(),
}));

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
  default: ({ href, children, ...props }: { href: string; children: ReactNode }) =>
    createElement("a", { href, ...props }, children),
}));

vi.mock("@/lib/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/api")>(),
  apiRequest: vi.fn(),
}));

import TalkPage from "@/app/talk/page";
import { ApiError, apiRequest } from "@/lib/api";
import { OPEN_CONVERSATION_KEY, journalSourceId, needsNewJournalChat } from "@/lib/conversation";
import { DEFAULT_PREFERENCES, savePreferences } from "@/lib/preferences";
import type { Conversation, ConversationTurn, JournalEntry } from "@/lib/types";

const ENTRY_ID = "10000000-0000-4000-8000-000000000001";
const OLD_CHAT_ID = "20000000-0000-4000-8000-000000000001";
const NEW_CHAT_ID = "20000000-0000-4000-8000-000000000002";
const entry: JournalEntry = {
  id: ENTRY_ID, user_id: "owner", created_at: "2026-10-04T10:00:00Z",
  text: "Original private journal text must not be copied into the chat composer.",
};

function conversation(id: string, source: string | null = null): Conversation {
  return {
    id, user_id: "owner", created_at: entry.created_at, updated_at: entry.created_at,
    status: "open", llm_consent: true, retain_text: false, safety_mode: "normal",
    locale: "CA", prompt_version: "mock", mode: "ai", revision: 0,
    source_entry_id: source, source_entry_created_at: source ? entry.created_at : null,
  };
}

function turn(chat: Conversation): ConversationTurn {
  return {
    conversation: { ...chat, revision: (chat.revision ?? 0) + 1 },
    user_message: {
      id: "message-user", conversation_id: chat.id, role: "user", content: "Stored old thought.",
      created_at: entry.created_at, safety_mode: "normal",
    },
    assistant_message: {
      id: "message-assistant", conversation_id: chat.id, role: "assistant", content: "DELAYED_OLD_REPLY",
      created_at: entry.created_at, safety_mode: "normal",
    },
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

let root: Root;
let container: HTMLDivElement;
const requested = vi.mocked(apiRequest);

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
  savePreferences({ ...DEFAULT_PREFERENCES, llmConsent: true });
  navigation.query = `entry=${ENTRY_ID}`;
  navigation.replace.mockImplementation((url: string) => {
    navigation.query = url.split("?")[1] ?? "";
    navigation.listeners.forEach((listener) => listener());
  });
  requested.mockReset();
  requested.mockImplementation(async (path) => {
    if (path === "/v1/system/status") return { analysis_mode: "ai_configured" };
    if (path === `/v1/journal/entries/${ENTRY_ID}`) return entry;
    if (path === `/v1/conversations/${OLD_CHAT_ID}`) return { conversation: conversation(OLD_CHAT_ID), messages: [] };
    if (path === `/v1/conversations/${OLD_CHAT_ID}/close`) return { ...conversation(OLD_CHAT_ID), status: "closed" };
    throw new Error(`Unexpected request ${path}`);
  });
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
  await act(async () => { root.render(createElement(TalkPage)); });
}

function button(label: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((item) => item.textContent?.trim() === label);
  expect(found, `Button '${label}' should be visible`).toBeTruthy();
  return found!;
}

async function click(label: string) {
  await act(async () => { button(label).click(); });
}

async function writeDraft(text: string) {
  const input = container.querySelector("textarea")!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!.call(input, text);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

async function submit() {
  await act(async () => {
    container.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
  });
}

describe("explicit journal selection", () => {
  test("a valid URL selects one identity and unrelated existing context requires a new chat", () => {
    expect(journalSourceId(ENTRY_ID.toUpperCase())).toBe(ENTRY_ID);
    expect(journalSourceId("all-entries")).toBeNull();
    expect(needsNewJournalChat(conversation(OLD_CHAT_ID), ENTRY_ID)).toBe(true);
    expect(needsNewJournalChat(conversation(OLD_CHAT_ID, ENTRY_ID), ENTRY_ID)).toBe(false);
  });

  test("entry navigation never replaces an unrelated open chat without confirmation", async () => {
    window.localStorage.setItem(OPEN_CONVERSATION_KEY, OLD_CHAT_ID);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    await mount();
    expect(container.textContent).toContain("Your current chat has different context");
    expect(container.textContent).not.toContain("Using your selected journal entry");
    await writeDraft("Keep this unsent draft.");
    await click("Use this entry in a new AI chat");
    expect(confirm).toHaveBeenCalledOnce();
    expect(container.querySelector("textarea")!.value).toBe("Keep this unsent draft.");
    expect(window.localStorage.getItem(OPEN_CONVERSATION_KEY)).toBe(OLD_CHAT_ID);
    expect(requested.mock.calls.some(([path]) => path.endsWith("/close"))).toBe(false);
  });

  test("confirmed source chat sends the UUID only and labels the original entry", async () => {
    const newChat = conversation(NEW_CHAT_ID, ENTRY_ID);
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === "/v1/conversations") return newChat;
      if (path === `/v1/conversations/${NEW_CHAT_ID}/messages`) return turn(newChat);
      return fallback(path, options);
    });
    await mount();
    expect(container.querySelector("textarea")!.disabled).toBe(true);
    await click("Use this entry in a new AI chat");
    expect(container.textContent).toContain("Using your selected journal entry");
    expect(container.querySelector(`a[href='/journal?entry=${ENTRY_ID}']`)).toBeTruthy();
    expect(container.textContent).not.toContain(entry.text);
    await writeDraft("What stood out was the silence.");
    await submit();
    const create = requested.mock.calls.find(([path]) => path === "/v1/conversations")!;
    expect(JSON.parse(String(create[1]!.body))).toMatchObject({ source_entry_id: ENTRY_ID, llm_consent: true });
    expect(String(create[1]!.body)).not.toContain(entry.text);
    expect(navigation.query).toContain(`c=${NEW_CHAT_ID}`);
    expect(navigation.query).toContain(`entry=${ENTRY_ID}`);
  });

  test("unavailable AI offers useful guided prompts without forwarding the entry", async () => {
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === "/v1/system/status") return { analysis_mode: "local_only" };
      if (path === "/v1/conversations") return { ...conversation(NEW_CHAT_ID), mode: "guided", llm_consent: false };
      if (path === `/v1/conversations/${NEW_CHAT_ID}/messages`) return turn(conversation(NEW_CHAT_ID));
      return fallback(path, options);
    });
    await mount();
    expect(button("Use this entry in a new AI chat").disabled).toBe(true);
    expect(container.textContent).toContain("your entry will not be sent");
    await click("Start guided chat without sending the entry");
    expect(container.textContent).toContain("what feels most important about it now?");
    await writeDraft("The pause in the meeting stayed with me.");
    await submit();
    const create = requested.mock.calls.find(([path]) => path === "/v1/conversations")!;
    const body = JSON.parse(String(create[1]!.body));
    expect(body.llm_consent).toBe(false);
    expect(body.source_entry_id).toBeUndefined();
  });

  test("missing or foreign entry cannot start a linked chat", async () => {
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path.startsWith("/v1/journal/entries/")) throw new ApiError("Journal entry not found", 404);
      return fallback(path, options);
    });
    await mount();
    expect(container.textContent).toContain("This journal entry is no longer available");
    expect(container.textContent).not.toContain("Use this entry in a new AI chat");
    expect(requested.mock.calls.some(([path]) => path === "/v1/conversations")).toBe(false);
  });

  test("a provider readiness refusal reveals the guided path after source selection", async () => {
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === "/v1/conversations") throw new ApiError(
        "Private AI help is unavailable. Start a guided chat without sending the entry.", 409,
      );
      return fallback(path, options);
    });
    await mount();
    await click("Use this entry in a new AI chat");
    await writeDraft("A private thought.");
    await submit();
    expect(container.textContent).toContain("AI help is off or unavailable");
    expect(button("Start guided chat without sending the entry").disabled).toBe(false);
    expect(container.querySelector("textarea")!.value).toBe("A private thought.");
    expect(requested.mock.calls.some(([path]) => path.endsWith("/messages"))).toBe(false);
  });

  test("a delayed reply cannot restore an ended chat or its open-chat identity", async () => {
    window.localStorage.setItem(OPEN_CONVERSATION_KEY, OLD_CHAT_ID);
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const delayed = deferred<ConversationTurn>();
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${OLD_CHAT_ID}/messages`) return delayed.promise;
      return fallback(path, options);
    });
    // Send in the old chat first; an entry link opened while its reply is pending must not
    // let that late reply revive the chat. (Sending under an open entry chooser is blocked.)
    navigation.query = `c=${OLD_CHAT_ID}`;
    await mount();
    await writeDraft("A thought for the old chat.");
    await submit();
    await act(async () => {
      navigation.query = `entry=${ENTRY_ID}`;
      navigation.listeners.forEach((listener) => listener());
    });
    expect(button("Use this entry in a new AI chat").disabled).toBe(true);
    // The end-chat control stays available while generation is pending. Its epoch
    // invalidates the outstanding response even if the provider later succeeds.
    await act(async () => { container.querySelector<HTMLButtonElement>("button[aria-label='Chat options']")!.click(); });
    await click("End this chat");
    await act(async () => { delayed.resolve(turn(conversation(OLD_CHAT_ID))); });
    expect(container.textContent).not.toContain("DELAYED_OLD_REPLY");
    expect(window.localStorage.getItem(OPEN_CONVERSATION_KEY)).toBeNull();
    expect(navigation.query).toBe("");
  });
});

describe("chat submission and recovery", () => {
  beforeEach(() => { navigation.query = `c=${OLD_CHAT_ID}`; });

  test("shows the submitted question before Luna's pending reply", async () => {
    const delayed = deferred<ConversationTurn>();
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${OLD_CHAT_ID}`) {
        const earlier = turn(conversation(OLD_CHAT_ID));
        earlier.assistant_message.content = "The earlier reply.";
        return { conversation: earlier.conversation, messages: [earlier.user_message, earlier.assistant_message] };
      }
      if (path.endsWith("/messages")) return delayed.promise;
      return fallback(path, options);
    });
    await mount();
    await writeDraft("Please help me understand this.");
    await submit();
    const bubble = [...container.querySelectorAll(".msg.from-me")].find((node) => node.textContent?.includes("Please help me understand this."));
    expect(bubble?.textContent).toContain("Please help me understand this.");
    expect(bubble?.textContent).toContain("Sending");
    const lunaBubbles = container.querySelectorAll(".msg.from-luna");
    expect(lunaBubbles[0].compareDocumentPosition(bubble!)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(bubble?.compareDocumentPosition(lunaBubbles[1]))
      .toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    const completed = turn(conversation(OLD_CHAT_ID));
    completed.user_message.id = "new-user";
    completed.user_message.content = "Please help me understand this.";
    completed.assistant_message.id = "new-assistant";
    await act(async () => { delayed.resolve(completed); });
    expect(container.textContent).not.toContain("Sending");
    expect([...container.querySelectorAll(".msg")].map((node) => node.textContent)).toEqual([
      "You said: Stored old thought.", "Luna said: The earlier reply.",
      "You said: Please help me understand this.", "Luna said: DELAYED_OLD_REPLY",
    ]);
  });

  test("a lost first-chat response retries the same creation receipt", async () => {
    navigation.query = "";
    const ids: string[] = [];
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === "/v1/conversations") {
        ids.push(JSON.parse(String(options!.body)).client_request_id);
        if (ids.length === 1) throw new Error("The creation response was lost.");
        return conversation(NEW_CHAT_ID);
      }
      if (path === `/v1/conversations/${NEW_CHAT_ID}/messages`) return turn(conversation(NEW_CHAT_ID));
      return fallback(path, options);
    });
    await mount();
    await writeDraft("My first thought.");
    await submit();
    await click("Try again");
    expect(ids).toHaveLength(2);
    expect(ids[1]).toBe(ids[0]);
  });

  test("an edited failed submission gets a fresh identity instead of replaying old text", async () => {
    const payloads: Array<{ client_message_id: string; text: string }> = [];
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path.endsWith("/messages")) {
        const payload = JSON.parse(String(options!.body));
        payloads.push(payload);
        if (payloads.length === 1) throw new Error("The response was lost.");
        const result = turn(conversation(OLD_CHAT_ID));
        result.user_message.content = payload.client_message_id === payloads[0].client_message_id
          ? payloads[0].text : payload.text;
        return result;
      }
      return fallback(path, options);
    });
    await mount();
    await writeDraft("The first wording.");
    await submit();
    expect(container.textContent).toContain("The response was lost.");
    await writeDraft("The corrected wording.");
    await submit();
    expect(payloads[1].client_message_id).not.toBe(payloads[0].client_message_id);
    expect(container.querySelector(".msg.from-me")?.textContent).toContain("The corrected wording.");
  });

  test("retrying the original message preserves newer unsent wording in the composer", async () => {
    let sends = 0;
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path.endsWith("/messages")) {
        if (++sends === 1) throw new Error("The response was lost.");
        return turn(conversation(OLD_CHAT_ID));
      }
      return fallback(path, options);
    });
    await mount();
    await writeDraft("Stored old thought.");
    await submit();
    await writeDraft("New wording I have not sent.");
    const retry = [...container.querySelectorAll("button")].find((item) => /Try again|Retry original message/.test(item.textContent ?? ""))!;
    await act(async () => { retry.click(); });
    expect(container.querySelector("textarea")!.value).toBe("New wording I have not sent.");
  });

  test("retrying unchanged text reuses its identity and does not duplicate an already refreshed turn", async () => {
    const stored = turn(conversation(OLD_CHAT_ID));
    const ids: string[] = [];
    let resumeReads = 0;
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${OLD_CHAT_ID}`) {
        resumeReads += 1;
        return resumeReads === 1 ? { conversation: conversation(OLD_CHAT_ID), messages: [] }
          : { conversation: stored.conversation, messages: [stored.user_message, stored.assistant_message] };
      }
      if (path.endsWith("/messages")) {
        ids.push(JSON.parse(String(options!.body)).client_message_id);
        if (ids.length === 1) throw new Error("The response was lost.");
        return stored;
      }
      if (path.endsWith("/preference")) throw new ApiError("Refresh first", 409);
      return fallback(path, options);
    });
    await mount();
    await writeDraft("Stored old thought.");
    await submit();
    await click("Just talk"); // A canonical refresh may discover the already committed turn.
    await submit();
    expect(ids[1]).toBe(ids[0]);
    expect(container.querySelectorAll(".msg.from-me")).toHaveLength(1);
    expect(container.querySelectorAll(".msg.from-luna")).toHaveLength(1);
  });

  test("switching the conversation query invalidates its delayed reply", async () => {
    const delayed = deferred<ConversationTurn>();
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${OLD_CHAT_ID}/messages`) return delayed.promise;
      if (path === `/v1/conversations/${NEW_CHAT_ID}`) {
        const next = turn(conversation(NEW_CHAT_ID));
        next.assistant_message.content = "The selected other chat.";
        return { conversation: next.conversation, messages: [next.user_message, next.assistant_message] };
      }
      return fallback(path, options);
    });
    await mount();
    await writeDraft("For the previous chat only.");
    await submit();
    await act(async () => { navigation.replace(`/talk?c=${NEW_CHAT_ID}`); });
    expect(container.textContent).toContain("The selected other chat.");
    await act(async () => { delayed.resolve(turn(conversation(OLD_CHAT_ID))); });
    expect(container.textContent).not.toContain("DELAYED_OLD_REPLY");
    expect(navigation.query).toBe(`c=${NEW_CHAT_ID}`);
    expect(window.localStorage.getItem(OPEN_CONVERSATION_KEY)).toBe(NEW_CHAT_ID);
  });

  test("navigation also invalidates a delayed first-chat creation", async () => {
    navigation.query = "";
    const delayed = deferred<Conversation>();
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === "/v1/conversations") return delayed.promise;
      if (path === `/v1/conversations/${NEW_CHAT_ID}`) {
        return { conversation: conversation(NEW_CHAT_ID), messages: [] };
      }
      return fallback(path, options);
    });
    await mount();
    await writeDraft("Belongs to the first new chat.");
    await submit();
    await act(async () => { navigation.replace(`/talk?c=${NEW_CHAT_ID}`); });
    await act(async () => { delayed.resolve(conversation(OLD_CHAT_ID)); });
    expect(navigation.query).toBe(`c=${NEW_CHAT_ID}`);
    expect(window.localStorage.getItem(OPEN_CONVERSATION_KEY)).toBe(NEW_CHAT_ID);
    expect(requested.mock.calls.some(([path]) => path.endsWith("/messages"))).toBe(false);
  });

  test("a delayed conflict refresh cannot attach the previous chat's error or retry to another chat", async () => {
    const delayed = deferred<{ conversation: Conversation; messages: [] }>();
    let oldReads = 0;
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${OLD_CHAT_ID}` && ++oldReads > 1) return delayed.promise;
      if (path === `/v1/conversations/${OLD_CHAT_ID}/messages`) throw new ApiError("Previous chat conflict.", 409);
      if (path === `/v1/conversations/${NEW_CHAT_ID}`) return { conversation: conversation(NEW_CHAT_ID), messages: [] };
      return fallback(path, options);
    });
    await mount();
    await writeDraft("Private previous chat wording.");
    await submit();
    await act(async () => { navigation.replace(`/talk?c=${NEW_CHAT_ID}`); });
    await act(async () => { delayed.resolve({ conversation: conversation(OLD_CHAT_ID), messages: [] }); });
    expect(container.textContent).not.toContain("Private previous chat wording.");
    expect(container.textContent).not.toContain("Previous chat conflict.");
    expect([...container.querySelectorAll("button")].some((item) => /Try again|Retry original/.test(item.textContent ?? ""))).toBe(false);
    expect(navigation.query).toBe(`c=${NEW_CHAT_ID}`);
  });

  test("unsent drafts stay with their conversation when navigating between chats", async () => {
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${NEW_CHAT_ID}`) return { conversation: conversation(NEW_CHAT_ID), messages: [] };
      return fallback(path, options);
    });
    await mount();
    await writeDraft("Only for the first chat.");
    await act(async () => { navigation.replace(`/talk?c=${NEW_CHAT_ID}`); });
    expect(container.querySelector("textarea")!.value).toBe("");
    await writeDraft("Only for the second chat.");
    await act(async () => { navigation.replace(`/talk?c=${OLD_CHAT_ID}`); });
    expect(container.querySelector("textarea")!.value).toBe("Only for the first chat.");
  });

  test("returning to a committed delayed turn does not restore submitted wording as a fresh draft", async () => {
    const delayed = deferred<ConversationTurn>();
    let stored: ConversationTurn | null = null;
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${OLD_CHAT_ID}/messages`) {
        stored = turn(conversation(OLD_CHAT_ID));
        stored.user_message.client_message_id = JSON.parse(String(options!.body)).client_message_id;
        stored.user_message.content = "Already submitted wording.";
        return delayed.promise;
      }
      if (path === `/v1/conversations/${OLD_CHAT_ID}` && stored) {
        return { conversation: stored.conversation, messages: [stored.user_message, stored.assistant_message] };
      }
      if (path === `/v1/conversations/${NEW_CHAT_ID}`) return { conversation: conversation(NEW_CHAT_ID), messages: [] };
      return fallback(path, options);
    });
    await mount();
    await writeDraft("Already submitted wording.");
    await submit();
    await act(async () => { navigation.replace(`/talk?c=${NEW_CHAT_ID}`); });
    await act(async () => { delayed.resolve(stored!); });
    await act(async () => { navigation.replace(`/talk?c=${OLD_CHAT_ID}`); });
    expect(container.querySelector("textarea")!.value).toBe("");
    expect(container.querySelectorAll(".msg.from-me")).toHaveLength(1);
    expect(container.textContent).not.toContain("Delivery not confirmed");
  });

  test("an unresolved submitted turn returns with its original retry receipt", async () => {
    const delayed = deferred<ConversationTurn>();
    const ids: string[] = [];
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${OLD_CHAT_ID}/messages`) {
        ids.push(JSON.parse(String(options!.body)).client_message_id);
        return ids.length === 1 ? delayed.promise : turn(conversation(OLD_CHAT_ID));
      }
      if (path === `/v1/conversations/${NEW_CHAT_ID}`) return { conversation: conversation(NEW_CHAT_ID), messages: [] };
      return fallback(path, options);
    });
    await mount();
    await writeDraft("Possibly undelivered wording.");
    await submit();
    await act(async () => { navigation.replace(`/talk?c=${NEW_CHAT_ID}`); });
    await act(async () => { navigation.replace(`/talk?c=${OLD_CHAT_ID}`); });
    expect(container.querySelector("textarea")!.value).toBe("Possibly undelivered wording.");
    expect(container.textContent).toContain("Delivery not confirmed");
    await click("Try again");
    expect(ids[1]).toBe(ids[0]);
    await act(async () => { delayed.resolve(turn(conversation(OLD_CHAT_ID))); });
  });

  test("acceptance retries share a receipt within one chat and get a new receipt after switching", async () => {
    const commands: Array<{ chat: string; id: string }> = [];
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${OLD_CHAT_ID}` || path === `/v1/conversations/${NEW_CHAT_ID}`) {
        const chat = conversation(path.endsWith(OLD_CHAT_ID) ? OLD_CHAT_ID : NEW_CHAT_ID);
        // Legacy guided acceptance still closes/saves; AI now uses an open activity session.
        chat.mode = "guided";
        chat.card = {
          resource_intent: "ground", card_reason: "A small step.", actions: [],
          decision_preview: {
            decision_id: "decision", action_id: "action", propensity: 1,
            policy_name: "fixed", policy_version: "test", safe_action_ids: ["action"],
            explanation: "A small step.", selection_source: "policy", eligible_for_ope: true,
          },
        };
        return { conversation: chat, messages: [] };
      }
      if (path.endsWith("/accept")) {
        commands.push({ chat: path, id: JSON.parse(String(options!.body)).client_request_id });
        throw new Error("Acceptance delivery was not confirmed.");
      }
      return fallback(path, options);
    });
    await mount();
    await click("Let’s try it");
    await click("Let’s try it");
    await act(async () => { navigation.replace(`/talk?c=${NEW_CHAT_ID}`); });
    await click("Let’s try it");
    expect(commands[1].id).toBe(commands[0].id);
    expect(commands[2].chat).not.toBe(commands[0].chat);
    expect(commands[2].id).not.toBe(commands[0].id);
  });

  test("leaving the workspace prevents a late reply from rewriting its route", async () => {
    const delayed = deferred<ConversationTurn>();
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path.endsWith("/messages")) return delayed.promise;
      return fallback(path, options);
    });
    await mount();
    await writeDraft("A pending thought.");
    await submit();
    await act(async () => { root.render(null); });
    navigation.replace.mockClear();
    await act(async () => { delayed.resolve(turn(conversation(OLD_CHAT_ID))); });
    expect(navigation.replace).not.toHaveBeenCalled();
  });

  test("a transient resume failure preserves the chat identity and offers recovery", async () => {
    window.localStorage.setItem(OPEN_CONVERSATION_KEY, OLD_CHAT_ID);
    const fallback = requested.getMockImplementation()!;
    let reads = 0;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${OLD_CHAT_ID}` && ++reads === 1) throw new ApiError("Temporarily unavailable", 503);
      return fallback(path, options);
    });
    await mount();
    expect(window.localStorage.getItem(OPEN_CONVERSATION_KEY)).toBe(OLD_CHAT_ID);
    expect(container.querySelector("textarea")!.disabled).toBe(true);
    await click("Try loading chat again");
    expect(container.querySelector("textarea")!.disabled).toBe(false);
  });

  test("Enter while composing text does not send a partial message", async () => {
    await mount();
    await writeDraft("正在输入");
    await act(async () => {
      container.querySelector("textarea")!.dispatchEvent(new KeyboardEvent("keydown", {
        key: "Enter", isComposing: true, bubbles: true,
      }));
    });
    expect(requested.mock.calls.some(([path]) => path.endsWith("/messages"))).toBe(false);
    expect(container.querySelector("textarea")!.value).toBe("正在输入");
  });
});

describe("resource discovery handoff", () => {
  test("normal chat opens discovery without forwarding its private source or message", async () => {
    navigation.query = `c=${OLD_CHAT_ID}&entry=${ENTRY_ID}`;
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${OLD_CHAT_ID}`) {
        const existing = turn(conversation(OLD_CHAT_ID, ENTRY_ID));
        existing.user_message.content = "A private detail about my friend.";
        // AI chats search inline; the library link remains for guided chats.
        existing.conversation.mode = "guided";
        return { conversation: existing.conversation, messages: [existing.user_message, existing.assistant_message] };
      }
      return fallback(path, options);
    });
    await mount();
    const link = [...container.querySelectorAll("a")].find((item) => item.textContent === "Find resources")!;
    expect(link.getAttribute("href")).toBe("/discover");
    expect(window.localStorage.getItem(OPEN_CONVERSATION_KEY)).toBe(OLD_CHAT_ID);
    expect(requested.mock.calls.some(([path]) => path.includes("discovery"))).toBe(false);
  });

  test("an offered saved resource collection hands off only its allowlisted general goal", async () => {
    navigation.query = `c=${OLD_CHAT_ID}&entry=${ENTRY_ID}`;
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${OLD_CHAT_ID}`) {
        const existing = conversation(OLD_CHAT_ID, ENTRY_ID);
        existing.mode = "guided";
        existing.card = {
          resource_intent: "ground", card_reason: "A private explanation.", goal: "settle", actions: [],
          decision_preview: {
            decision_id: "decision", action_id: "action", propensity: 1,
            policy_name: "fixed", policy_version: "test", safe_action_ids: [],
            explanation: "A private explanation.", selection_source: "policy", eligible_for_ope: true,
          },
        };
        return { conversation: existing, messages: [] };
      }
      return fallback(path, options);
    });
    await mount();
    const link = [...container.querySelectorAll("a")].find((item) => item.textContent === "Search for other resources")!;
    expect(link.getAttribute("href")).toBe("/discover?goal=settle");
    expect(container.textContent).toContain("Saved resource collection");
    expect(requested.mock.calls.some(([path]) => path.includes("discovery"))).toBe(false);
  });

  test("support mode keeps web discovery outside the support flow", async () => {
    navigation.query = `c=${OLD_CHAT_ID}`;
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path === `/v1/conversations/${OLD_CHAT_ID}`) {
        return { conversation: { ...conversation(OLD_CHAT_ID), safety_mode: "support" }, messages: [] };
      }
      return fallback(path, options);
    });
    await mount();
    expect(container.querySelectorAll("a[href^='/discover']")).toHaveLength(0);
    expect(container.textContent).toContain("You don’t have to handle this alone.");
  });
});

describe("composer during a reply", () => {
  beforeEach(() => { navigation.query = `c=${OLD_CHAT_ID}`; });

  test("keeps focus and the draft field enabled while Luna replies, but blocks a second send", async () => {
    const delayed = deferred<ConversationTurn>();
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, options) => {
      if (path.endsWith("/messages")) return delayed.promise;
      return fallback(path, options);
    });
    await mount();
    const input = container.querySelector("textarea")!;
    await act(async () => { input.focus(); });
    await writeDraft("First thought.");
    await submit();
    // Disabling the field would drop focus and close a mobile keyboard after every message.
    expect(input.disabled).toBe(false);
    expect(input.readOnly).toBe(true);
    expect(document.activeElement).toBe(input);
    expect(container.querySelector("[role=status]")?.textContent).toContain("Luna is replying");
    await submit();
    expect(requested.mock.calls.filter(([path]) => path.endsWith("/messages"))).toHaveLength(1);
    const completed = turn(conversation(OLD_CHAT_ID));
    completed.user_message.content = "First thought.";
    await act(async () => { delayed.resolve(completed); });
    expect(input.readOnly).toBe(false);
    expect(document.activeElement).toBe(input);
  });
});

describe("AI chat search entry", () => {
  test("an AI chat offers one inline search instead of also linking to the separate library", async () => {
    navigation.query = `c=${OLD_CHAT_ID}`;
    await mount();
    expect(container.querySelectorAll("a[href^='/discover']")).toHaveLength(0);
  });
});

describe("entry chooser over an open chat", () => {
  test("the composer waits for a choice instead of sending into the unrelated chat", async () => {
    window.localStorage.setItem(OPEN_CONVERSATION_KEY, OLD_CHAT_ID);
    navigation.query = `entry=${ENTRY_ID}`;
    await mount();
    expect(container.textContent).toContain("Start a new chat to use this entry");
    // Typing here would go to the old chat without the entry while the screen offers the entry.
    expect(container.querySelector("textarea")!.disabled).toBe(true);
    await click("Keep current chat");
    expect(container.textContent).not.toContain("Start a new chat to use this entry");
    expect(container.querySelector("textarea")!.disabled).toBe(false);
  });
});
