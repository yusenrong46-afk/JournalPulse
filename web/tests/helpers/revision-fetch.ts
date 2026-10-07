/** Existing transport fixtures focus on the operation; the dedicated revision
 * tests exercise this extra read, cancellation, and immutable header binding. */
export function withDataRevisionPreflight(operation: typeof fetch): typeof fetch {
  return async (url, init) => {
    const path = new URL(String(url), window.location.origin).pathname;
    if (path === "/v1/account/data-revision" && (!init?.method || init.method === "GET")) {
      return new Response(JSON.stringify({ revision: 0 }));
    }
    return operation(url, init);
  };
}
