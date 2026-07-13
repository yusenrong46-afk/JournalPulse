import type { ActionPreview, AffectiveState, PreparedAnalysis, TargetState } from "./types";

const DATABASE = "journalpulse_private_v1";
const STORE = "encrypted_drafts";
const ACTIVE_DRAFT = "active_reflection";
const MAX_AGE_MS = 24 * 60 * 60 * 1000;

export type ReflectionDraft = {
  clientRequestId: string;
  step: number;
  text: string;
  situation: string;
  energy: string;
  socialContext: string;
  consentOverride: boolean | null;
  retainOverride: boolean | null;
  analysis: PreparedAnalysis | null;
  state: AffectiveState;
  target: TargetState;
  preview: ActionPreview | null;
  selectedAction: string;
};

type StoredDraft = {
  id: string;
  key: CryptoKey;
  iv?: number[];
  ciphertext?: number[];
  updatedAt?: number;
};

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: "id" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function readRecord(database: IDBDatabase, id: string): Promise<StoredDraft | undefined> {
  return new Promise((resolve, reject) => {
    const request = database.transaction(STORE).objectStore(STORE).get(id);
    request.onsuccess = () => resolve(request.result as StoredDraft | undefined);
    request.onerror = () => reject(request.error);
  });
}

async function writeRecord(database: IDBDatabase, value: StoredDraft): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = database.transaction(STORE, "readwrite").objectStore(STORE).put(value);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

async function encryptionKey(database: IDBDatabase): Promise<CryptoKey> {
  const stored = await readRecord(database, "device_key");
  if (stored?.key) return stored.key;
  const key = await crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, false, [
    "encrypt",
    "decrypt",
  ]);
  await writeRecord(database, { id: "device_key", key });
  return key;
}

export async function saveReflectionDraft(draft: ReflectionDraft): Promise<void> {
  const database = await openDatabase();
  try {
    const key = await encryptionKey(database);
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const plaintext = new TextEncoder().encode(JSON.stringify(draft));
    const ciphertext = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, plaintext);
    await writeRecord(database, {
      id: ACTIVE_DRAFT,
      key,
      iv: Array.from(iv),
      ciphertext: Array.from(new Uint8Array(ciphertext)),
      updatedAt: Date.now(),
    });
  } finally {
    database.close();
  }
}

export async function loadReflectionDraft(): Promise<ReflectionDraft | null> {
  const database = await openDatabase();
  try {
    const stored = await readRecord(database, ACTIVE_DRAFT);
    if (!stored?.iv || !stored.ciphertext || !stored.updatedAt) return null;
    if (Date.now() - stored.updatedAt > MAX_AGE_MS) {
      await clearReflectionDraft(database);
      return null;
    }
    const key = await encryptionKey(database);
    const plaintext = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: new Uint8Array(stored.iv) },
      key,
      new Uint8Array(stored.ciphertext),
    );
    return JSON.parse(new TextDecoder().decode(plaintext)) as ReflectionDraft;
  } catch {
    return null;
  } finally {
    database.close();
  }
}

export async function clearReflectionDraft(existingDatabase?: IDBDatabase): Promise<void> {
  const database = existingDatabase ?? (await openDatabase());
  try {
    await new Promise<void>((resolve, reject) => {
      const request = database.transaction(STORE, "readwrite").objectStore(STORE).delete(ACTIVE_DRAFT);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  } finally {
    if (!existingDatabase) database.close();
  }
}
