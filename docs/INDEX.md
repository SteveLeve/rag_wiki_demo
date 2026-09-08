# Documentation Index

Central hub for all RAG Wiki Demo documentation, organized by audience and purpose.

## 👤 For New Users

Start here if you're new to RAG or this project:

- **[Getting Started](./user-guides/getting-started.md)** - Quick onboarding checklist (5 min read)
- **[Learning Roadmap](./learning-paths/learning-roadmap.md)** - Choose your learning path (Path A/B/C with time estimates)
- **[Quick Reference](./user-guides/quick-reference.md)** - Decision guide for storage backends
- **[PostgreSQL Setup](./user-guides/postgres-setup.md)** - Database configuration and troubleshooting

## 🎓 For Learners

Deepen your understanding of RAG concepts and techniques:

### Foundational Knowledge
- **[RAG Concepts](./learning-paths/concepts.md)** - Core RAG theory and architecture
- **[Advanced Concepts](./learning-paths/advanced-concepts.md)** - Production RAG techniques
- **[Evaluation Concepts](./learning-paths/evaluation-concepts.md)** - How to measure RAG quality

### Learning Notebooks
- **[foundation/](../foundation/)** - RAG fundamentals (start here!)
- **[intermediate/](../intermediate/)** - Registry patterns and model comparison
- **[advanced-techniques/](../advanced-techniques/)** - Specialized improvements
- **[evaluation-lab/](../evaluation-lab/)** - Measurement and comparison

## 🛠️ For Contributors & Agents

- **[AGENTS.md](../AGENTS.md)** - Repository conventions: the teaching-copy rule, the canonical
  alias invariant, and how to verify changes. Read this before editing notebooks or `src/ragkit/`.

### Running the checks

| Check | Command |
|---|---|
| Unit tests, no services needed | `RAG_FAKE_MODELS=1 pytest -m "not postgres"` |
| Full suite | `pytest` (needs PostgreSQL; see [postgres-setup](./user-guides/postgres-setup.md)) |
| Notebook conventions | `python scripts/nb_lint.py` |
| Documentation links | `python scripts/check_links.py` |
| Model catalog matches reality | `python scripts/preflight.py` |

---

## 📊 Directory Structure

```
docs/
├── user-guides/              # For all users - practical setup guides
│   ├── getting-started.md
│   ├── quick-reference.md
│   └── postgres-setup.md
│
└── learning-paths/           # For learners - conceptual and educational
    ├── learning-roadmap.md
    ├── concepts.md
    ├── advanced-concepts.md
    └── evaluation-concepts.md
```

Point-in-time phase reports, release notes, and validation summaries were removed in September 2026.
They described a January snapshot and had begun to read as current status. `git log` retains them.

---

## 🔗 Quick Links

| Need | Link |
|------|------|
| I'm new here | [Getting Started](./user-guides/getting-started.md) |
| Setting up PostgreSQL | [Postgres Setup](./user-guides/postgres-setup.md) |
| Choosing a learning path | [Learning Roadmap](./learning-paths/learning-roadmap.md) |
| Understanding RAG | [Concepts](./learning-paths/concepts.md) |
| Measuring quality | [Evaluation Concepts](./learning-paths/evaluation-concepts.md) |
| Contributing / agent onboarding | [AGENTS.md](../AGENTS.md) |
