import { act, createElement, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

const fixture = vi.hoisted(() => ({ getSupabase: vi.fn(), replace: vi.fn() }));
vi.mock('@/lib/supabase', () => ({ getSupabase: fixture.getSupabase }));
vi.mock('next/navigation', () => {
  const router = { replace: fixture.replace };
  const query = new URLSearchParams();
  return { useRouter: () => router, useSearchParams: () => query };
});
vi.mock('next/link', () => ({
  default: ({ href, children, scroll, ...props }: { href: string; children: ReactNode; scroll?: boolean }) => {
    void scroll;
    return createElement('a', { href, ...props }, children);
  },
}));

import JournalPage from '@/app/journal/page';
import MePage from '@/app/me/page';
import { DEFAULT_PREFERENCES, savePreferences } from '@/lib/preferences';
import { activateBrowserAccount } from '@/lib/account-storage';
import { invalidateAccountDataRequests } from '@/lib/account-data';
import { saveJournalEntry } from '@/lib/journal';
import { readOpenConversationId, writeOpenConversationId } from '@/lib/conversation';

const userId = '10000000-0000-4000-8000-000000000001';
const client = { auth: { getSession: async () => ({ data: { session: { access_token: 'synthetic-token-only', user: { id: userId } } } }) } };
let container: HTMLDivElement;
let root: Root;
const calls: { method: string; path: string; text?: string; requestId?: string; dataRevision?: string | null }[] = [];
let entries: { id: string; user_id: string; created_at: string; text: string }[];
let firstSaveResponse: Promise<Response> | null = null;
let deleteResponse: Promise<Response> | null = null;
let deleteFails = false;
let rejectSaveAfterRemoteErasure = false;
let serverDataRevision = 0;

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  activateBrowserAccount(userId);
  localStorage.clear();
  savePreferences({ ...DEFAULT_PREFERENCES, onboarded: true });
  calls.length = 0;
  firstSaveResponse = null;
  deleteResponse = null;
  deleteFails = false;
  rejectSaveAfterRemoteErasure = false;
  serverDataRevision = 0;
  entries = [{ id: '30000000-0000-4000-8000-000000000001', user_id: userId, created_at: '2026-10-06T10:00:00Z', text: 'Older synthetic journal entry' }];
  fixture.getSupabase.mockReset().mockResolvedValue(client);
  vi.stubGlobal('fetch', vi.fn(async (url: string, options?: RequestInit) => {
    const method = options?.method ?? 'GET';
    const target = new URL(String(url), window.location.origin);
    const path = target.pathname + target.search;
    const body = options?.body ? JSON.parse(String(options.body)) : null;
    calls.push({ method, path, ...(body?.text ? { text: body.text, requestId: body.client_request_id,
      dataRevision: new Headers(options?.headers).get('X-JournalPulse-Data-Revision') } : {}) });
    let payload: unknown;
    if (path === '/v1/account/data-revision') payload = { revision: serverDataRevision };
    else if (path === '/v1/journal/entries?limit=50&offset=0') payload = { items: [...entries], limit: 50, offset: 0 };
    else if (path === '/v1/account/data' && method === 'DELETE') {
      if (deleteFails) return new Response(JSON.stringify({ detail: 'Synthetic unavailable deletion' }), { status: 503 });
      serverDataRevision += 1;
      payload = { deleted_records: entries.length }; entries = [];
      if (deleteResponse) return deleteResponse;
    } else if (path === '/v1/journal/entries' && method === 'POST') {
      if (rejectSaveAfterRemoteErasure) {
        rejectSaveAfterRemoteErasure = false;
        serverDataRevision += 1;
        entries = [];
        return new Response(JSON.stringify({ detail: 'Saved data was erased on another device.' }), { status: 409 });
      }
      const entry = { id: body.client_request_id, text: body.text, user_id: userId, created_at: '2026-10-06T10:05:00Z' };
      entries.push(entry); payload = entry;
      if (firstSaveResponse) {
        const delayed = firstSaveResponse; firstSaveResponse = null;
        return delayed;
      }
    } else throw new Error(`Unexpected synthetic path ${method} ${path}`);
    return new Response(JSON.stringify(payload), { headers: { 'Content-Type': 'application/json' } });
  }));
  container = document.createElement('div'); document.body.append(container);
  root = createRoot(container);
});

test('confirmed deletion prevents replay after a committed save loses its response', async () => {
  await act(async () => { root.render(createElement(JournalPage)); });
  await write(container.querySelector('textarea')!, 'Synthetic saved text that deletion should remove');
  let rejectResponse!: (reason: Error) => void;
  firstSaveResponse = new Promise<Response>((_resolve, reject) => { rejectResponse = reject; });
  await act(async () => { button('Save entry').click(); });
  expect(entries).toHaveLength(2);
  await act(async () => { root.render(createElement(MePage)); });
  await write(container.querySelector('input[autocomplete="off"]')!, 'delete my journal');
  await act(async () => { button('Delete my journal').click(); });
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 10)); });
  expect(container.textContent).toContain('Done. 2 saved items were deleted.');
  expect(entries).toHaveLength(0);
  await act(async () => {
    rejectResponse(new TypeError('Synthetic lost first response'));
    await new Promise((resolve) => setTimeout(resolve, 400));
  });
  const mutations = calls.filter((call) => call.method !== 'GET');
  expect(mutations.map((call) => call.method)).toEqual(['POST', 'DELETE']);
  expect(entries).toHaveLength(0);
});
afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove(); vi.unstubAllGlobals();
});

async function write(element: HTMLInputElement | HTMLTextAreaElement, value: string) {
  const prototype = element instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  await act(async () => {
    Object.getOwnPropertyDescriptor(prototype, 'value')!.set!.call(element, value);
    element.dispatchEvent(new Event('input', { bubbles: true }));
  });
}
function button(text: string) {
  const control = [...container.querySelectorAll('button')].find((item) => item.textContent === text);
  expect(control, `Expected ${text}`).toBeTruthy(); return control!;
}

test('confirmed deletion prevents a pending-authentication journal save from starting', async () => {
  await act(async () => { root.render(createElement(JournalPage)); });
  await write(container.querySelector('textarea')!, 'Synthetic writing queued before delete');
  let resolveAuth!: (value: typeof client) => void;
  fixture.getSupabase.mockReturnValueOnce(new Promise<typeof client>((resolve) => { resolveAuth = resolve; }));
  await act(async () => { button('Save entry').click(); });
  expect(calls.some((call) => call.method === 'POST')).toBe(false);
  await act(async () => { root.render(createElement(MePage)); });
  await write(container.querySelector('input[autocomplete="off"]')!, 'delete my journal');
  await act(async () => { button('Delete my journal').click(); });
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 10)); });
  expect(container.textContent).toContain('Done. 1 saved items were deleted.');
  expect(entries).toHaveLength(0);
  await act(async () => { resolveAuth(client); });
  const mutations = calls.filter((call) => call.method !== 'GET');
  expect(mutations.map((call) => call.method)).toEqual(['DELETE']);
  expect(entries).toHaveLength(0);
  await act(async () => { root.render(createElement(JournalPage)); });
  expect(container.textContent).not.toContain('Synthetic writing queued before delete');
  await write(container.querySelector('textarea')!, 'A new intentional entry after deletion');
  await act(async () => { button('Save entry').click(); });
  expect(entries.map((entry) => entry.text)).toEqual(['A new intentional entry after deletion']);
});

test('a failed deletion still cancels older work, preserves local references, and allows a fresh save', async () => {
  const resumeId = '10000000-0000-4000-8000-000000000009';
  writeOpenConversationId(resumeId);
  await act(async () => { root.render(createElement(JournalPage)); });
  await write(container.querySelector('textarea')!, 'Writing submitted before failed deletion');
  let resolveAuth!: (value: typeof client) => void;
  fixture.getSupabase.mockReturnValueOnce(new Promise<typeof client>((resolve) => { resolveAuth = resolve; }));
  await act(async () => { button('Save entry').click(); });
  await act(async () => { root.render(createElement(MePage)); });
  await write(container.querySelector('input[autocomplete="off"]')!, 'delete my journal');
  deleteFails = true;
  await act(async () => { button('Delete my journal').click(); });
  expect(container.textContent).toContain('Deletion didn’t finish');
  expect(container.textContent).not.toContain('saved items were deleted');
  expect(readOpenConversationId()).toBe(resumeId);
  await act(async () => { resolveAuth(client); });
  expect(calls.filter((call) => call.method !== 'GET').map((call) => call.method)).toEqual(['DELETE']);
  expect(entries).toHaveLength(1);
  await act(async () => { root.render(createElement(JournalPage)); });
  await write(container.querySelector('textarea')!, 'A fresh intentional save after failed deletion');
  await act(async () => { button('Save entry').click(); });
  expect(entries.map((entry) => entry.text)).toContain('A fresh intentional save after failed deletion');
});

test('successful deletion also cancels a save started while the deletion response was pending', async () => {
  let resolveDeletion!: (value: Response) => void;
  deleteResponse = new Promise<Response>((resolve) => { resolveDeletion = resolve; });
  await act(async () => { root.render(createElement(MePage)); });
  await write(container.querySelector('input[autocomplete="off"]')!, 'delete my journal');
  await act(async () => { button('Delete my journal').click(); });
  let resolveAuth!: (value: typeof client) => void;
  fixture.getSupabase.mockReturnValueOnce(new Promise<typeof client>((resolve) => { resolveAuth = resolve; }));
  const concurrentSave = saveJournalEntry('Writing submitted during deletion', crypto.randomUUID()).catch((reason: Error) => reason.message);
  await act(async () => {
    resolveDeletion(new Response(JSON.stringify({ deleted_records: 1 })));
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
  expect(container.textContent).toContain('Done. 1 saved items were deleted.');
  resolveAuth(client);
  expect(await concurrentSave).toContain('cancelled');
  expect(calls.filter((call) => call.method !== 'GET').map((call) => call.method)).toEqual(['DELETE']);
  expect(entries).toHaveLength(0);
});

test('ordinary navigation still lets an intentionally submitted save finish in the background', async () => {
  await act(async () => { root.render(createElement(JournalPage)); });
  await write(container.querySelector('textarea')!, 'An intentional background save');
  let resolveAuth!: (value: typeof client) => void;
  fixture.getSupabase.mockReturnValueOnce(new Promise<typeof client>((resolve) => { resolveAuth = resolve; }));
  await act(async () => { button('Save entry').click(); });
  await act(async () => { root.render(createElement(MePage)); });
  await act(async () => { resolveAuth(client); });
  expect(calls.filter((call) => call.method !== 'GET').map((call) => call.method)).toEqual(['POST']);
  expect(entries.map((entry) => entry.text)).toContain('An intentional background save');
  expect(container.textContent).toContain('Your space');
});

test('an explicit save in the surviving editor uses a new receipt after another tab erases data', async () => {
  await act(async () => { root.render(createElement(JournalPage)); });
  await write(container.querySelector('textarea')!, 'Writing deliberately saved again after erasure');
  let rejectResponse!: (reason: Error) => void;
  firstSaveResponse = new Promise<Response>((_resolve, reject) => { rejectResponse = reject; });
  await act(async () => { button('Save entry').click(); });
  const originalId = calls.find((call) => call.method === 'POST')!.requestId;
  await act(async () => {
    entries = [];
    invalidateAccountDataRequests();
    rejectResponse(new TypeError('Synthetic lost response after remote erasure'));
  });
  expect(container.querySelector('textarea')?.value).toBe('Writing deliberately saved again after erasure');
  await act(async () => { button('Save entry').click(); });
  const saves = calls.filter((call) => call.method === 'POST');
  expect(saves).toHaveLength(2);
  expect(saves[1].requestId).not.toBe(originalId);
  expect(entries).toHaveLength(1);
});

test('a conflict without a browser notice retains writing and renews its receipt only on the next explicit Save', async () => {
  await act(async () => { root.render(createElement(JournalPage)); });
  await write(container.querySelector('textarea')!, 'Deliberately saved after another device erases data');
  rejectSaveAfterRemoteErasure = true;
  await act(async () => { button('Save entry').click(); });
  expect(container.textContent).toContain('Saved data was erased on another device.');
  expect(container.querySelector('textarea')?.value).toBe('Deliberately saved after another device erases data');
  expect(calls.filter((call) => call.method === 'POST')).toHaveLength(1);
  expect(entries).toHaveLength(0);
  await act(async () => { button('Save entry').click(); });
  const saves = calls.filter((call) => call.method === 'POST');
  expect(saves).toHaveLength(2);
  expect(saves.map((call) => call.dataRevision)).toEqual(['0', '1']);
  expect(saves[1].requestId).not.toBe(saves[0].requestId);
  expect(entries.map((entry) => entry.text)).toEqual(['Deliberately saved after another device erases data']);
});
