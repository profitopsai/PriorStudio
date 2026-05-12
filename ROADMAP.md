# PriorStudio Roadmap

## v0.1 — Scaffolding ✅
- [x] Repo structure, license, docs skeleton
- [x] FM project template with examples for every artifact type
- [x] JSON Schemas for prior / model / eval / run / initiative
- [x] CONTRIBUTING.md with PR conventions

## v0.2 — CLI ✅
- [x] `priorstudio init/validate/lint/list-artifacts/run/version`
- [x] Cross-reference checks (run→prior/model/eval, dir/id match, bibkeys)

## v0.3 — Static Studio ✅
- [x] Static-site generator (priorstudio-studio package)
- [x] `studio build` and `studio serve`

## v0.4 — Compute & tracking ✅
- [x] core package: Prior/Model/Eval/Run, registry, loaders
- [x] Built-in PyTorch blocks
- [x] Local PFN training loop
- [x] Compute adapters: local works, vast/modal/runpod/hf_spaces scaffolded
- [x] Tracking adapters: local + W&B + MLflow
- [x] Tests + GitHub Actions CI

## v0.5 — Web app + database ⚙️
- [x] Prisma schema for all entities (User, Org, OrgMember, Project, Prior, Model, Eval, Run, Initiative, LiteratureEntry, ApiToken)
- [x] NestJS API with JWT + API token auth, multi-tenant guards (ProjectAccessGuard + role-based)
- [x] CRUD endpoints for all artifact types
- [x] Export endpoint (project → tar.gz of canonical FM file layout)
- [x] Angular shell with login/register, project list, project nav
- [x] Priors fully wired UI (list / create / edit / delete with parameter table, code editor, citations)
- [x] Roadmap markdown editor with preview
- [x] Initiatives list + create
- [x] Docker Compose for Postgres
- [x] Seed script (demo workspace + demo project)
- [ ] Models / Evals / Runs / Literature UIs (API works; editors stubbed)
- [ ] CLI ↔ API integration (`priorstudio login`, `priorstudio pull`, `priorstudio push`)
- [ ] Full SSH/scp orchestration for Vast adapter
- [ ] PyPI release for python packages

## v0.6 — Polish
- [ ] Models / Evals / Runs / Literature editor UIs
- [ ] Diff view between artifact versions
- [ ] Run submission from the web UI (kicks off compute adapter on backend worker)
- [ ] Run live status (websocket / SSE)
- [ ] Comments / @mentions on artifacts
- [ ] First external project shipped on it

## Toward v1.0 (no commitment yet)
v1.0 means schemas frozen and the platform is stable.
- [ ] Schema versioning + migration story
- [ ] Plugin system for custom artifact types
- [ ] At least 3 external PFN projects in production on it
- [ ] Hosted version at priorstudio.dev
- [ ] Documentation site
