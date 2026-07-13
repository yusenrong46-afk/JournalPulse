import "fake-indexeddb/auto";

import { webcrypto } from "node:crypto";

Object.defineProperty(globalThis, "crypto", {
  configurable: true,
  value: webcrypto,
});

const values = new Map<string, string>();
const memoryStorage: Storage = {
  get length() {
    return values.size;
  },
  clear: () => values.clear(),
  getItem: (key) => values.get(key) ?? null,
  key: (index) => Array.from(values.keys())[index] ?? null,
  removeItem: (key) => values.delete(key),
  setItem: (key, value) => values.set(key, String(value)),
};
Object.defineProperty(window, "localStorage", {
  configurable: true,
  value: memoryStorage,
});
