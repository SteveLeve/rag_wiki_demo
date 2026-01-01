# Documentation Index

Central hub for all RAG Wiki Demo documentation, organized by audience and purpose.

## 👤 For New Users

Start here if you're new to RAG or this project:

- **[Getting Started](./user-guides/getting-started.md)** - Quick onboarding checklist (5 min read)
- **[Learning Roadmap](./learning-paths/learning-roadmap.md)** - Choose your learning path (Path A/B/C with time estimates)
- **[Quick Reference](./user-guides/quick-reference.md)** - Decision guide for storage backends

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

## 🛠️ For Developers & Contributors

Documentation for maintaining and extending the project:

### Getting Help
- **[PostgreSQL Setup](./user-guides/postgres-setup.md)** - Detailed database configuration and troubleshooting

### Development Artifacts

#### Testing & Quality
- **[Testing Guide](./development/testing/testing-guide.md)** - How to write and run tests
- **[Testing Summary](./development/testing/testing-summary.md)** - Current test coverage and results
- **[Evaluation Guide](./development/testing/evaluation-guide.md)** - RAG evaluation methodology

#### Reports & Analysis
- **[Validation Report](./development/reports/notebook-validation-report.md)** - Current notebook execution status
- **[Fixes Report](./development/reports/notebook-fixes-report.md)** - Known issues and resolutions
- **[Cross-Reference Report](./development/reports/cross-reference-report.md)** - Documentation link validation
- **[Execution Summary](./development/reports/execution-summary.md)** - Test execution results
- **[Verification Report](./development/reports/verification-report.md)** - System verification status

#### Release Information
- **[Release Notes](./development/releases/release-notes.md)** - What's new in each release
- **[Changelog](./development/releases/changelog.md)** - Detailed change history
- **[Version History](./development/releases/version-history.md)** - All version summaries

#### Implementation Notes
- **[Implementation Summary](./development/implementation/implementation-summary.md)** - Detailed implementation specifics
- **[Implementation Progress](./development/implementation/implementation-progress.md)** - Feature completion status
- **[Enhancement Summary](./development/implementation/enhancement-summary.md)** - New features and improvements

---

## 📊 Directory Structure

```
docs/
├── user-guides/              # For all users - practical setup guides
│   ├── getting-started.md
│   ├── quick-reference.md
│   └── postgres-setup.md
│
├── learning-paths/           # For learners - conceptual and educational
│   ├── learning-roadmap.md
│   ├── concepts.md
│   ├── advanced-concepts.md
│   └── evaluation-concepts.md
│
└── development/              # For developers - maintenance and analysis
    ├── testing/              # QA and testing
    │   ├── testing-guide.md
    │   ├── testing-summary.md
    │   └── evaluation-guide.md
    │
    ├── reports/              # Analysis and validation
    │   ├── notebook-validation-report.md
    │   ├── notebook-fixes-report.md
    │   ├── cross-reference-report.md
    │   ├── execution-summary.md
    │   └── verification-report.md
    │
    ├── releases/             # Version and release info
    │   ├── release-notes.md
    │   ├── changelog.md
    │   └── version-history.md
    │
    └── implementation/       # Technical details
        ├── implementation-summary.md
        ├── implementation-progress.md
        └── enhancement-summary.md
```

---

## 🔗 Quick Links

| Need | Link |
|------|------|
| I'm new here | [Getting Started](./user-guides/getting-started.md) |
| Setting up PostgreSQL | [Postgres Setup](./user-guides/postgres-setup.md) |
| Choosing a learning path | [Learning Roadmap](./learning-paths/learning-roadmap.md) |
| Understanding RAG | [Concepts](./learning-paths/concepts.md) |
| What's new? | [Release Notes](./development/releases/release-notes.md) |
| Project status | [Implementation Progress](./development/implementation/implementation-progress.md) |
| Current issues | [Validation Report](./development/reports/notebook-validation-report.md) |

---

**Last updated:** 2026-01-01
