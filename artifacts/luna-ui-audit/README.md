# Luna UI audit evidence (2026-10-05)

Report: [`docs/LUNA_UI_AUDIT_2026-10-05.md`](../../docs/LUNA_UI_AUDIT_2026-10-05.md).

- `before/`: starting-point check logs, `BASELINE.md` (toolchain, flaky checks), and `screens/`
  from the original UI.
- `after/`: final logs for the same checks, `ui-tour-axe.log`, and `screens/` from the redesigned
  UI. The images were captured with the same `web/tests/integration/ui-tour.spec.ts`.
- `fixes/`: the new regression tests run against the original code (F1, F2), and the dependency
  advisory recheck (F6).
- `patches/`: one code patch per commit after `e0bceab`. Evidence files are excluded.

Screens use the local stack's stand-ins for Luna, Brave and sign-in. No real writing, credentials
or paid calls appear in any file here.
