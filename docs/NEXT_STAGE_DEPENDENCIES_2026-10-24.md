# Bounded dependency advisory repair

REQ-BASE / REQ-REL. The locked install reported six affected dependency packages:
Next (critical), Vitest/mocker (moderate), Vite, Undici and Zod (high). This was a
new concrete finding from the current install, not a reason to upgrade the stack.
Original `npm-audit.json` is preserved.

Primary Next maintainer advisories:
- https://github.com/vercel/next.js/security/advisories/GHSA-2464-fhr4-w9xw
- https://github.com/vercel/next.js/security/advisories/GHSA-673q-mjm6-r5xr
- https://github.com/vercel/next.js/security/advisories/GHSA-p3m5-59mp-jv5r

The reported Next vulnerabilities concern image optimization, `use cache` with
multiple typed arrays and Server Actions. This source serves a static Next export
through FastAPI. Searches found no `next/image`, `use cache` or Server Action in
app/components/lib. That is a source-based exposure assessment, not hosted
penetration testing or a claim that vulnerable packages are safe in other modes.

Next and its matching ESLint config move from16.3.8 to16.4.3. The remaining concrete
advisories receive semver-compatible dependency fixes without `--force`. The first
selective `npm update` failed inside npm (`edgesOut`); it is preserved and was
followed by `npm audit fix`, which hit the same npm10 peer-resolution bug.
A temporary npm12.2.1 runtime (engine supports Node22.23.3) completed the selective
update of Vitest/mocker, Vite, Undici and Zod. No global npm/runtime change was made.
The final full npm audit reports zero known advisories; fresh npm ci and affected
checks verify the resulting lockfile. No production deployment or
application architecture change is part of this repair. Consult the final audit
and repeated frontend checks for the actual resulting package set and status.
The earlier frontend checks refer to the original lockfile and are superseded by
`*-release.log` checks for this final lockfile.

The Python inference source, catalog, model requests and v3 evaluation freeze are
unchanged by this frontend-only repair. The final source manifest includes the
new frontend lockfile identity. Advisory absence from a particular audit result
is not proof that the complete application has no security defects.
