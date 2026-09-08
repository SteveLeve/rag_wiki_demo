"""The model catalog, and the one function allowed to produce an alias.

An embedding model carries two distinct names, and conflating them is what broke
this repository before September 2026:

  ollama_tag   what you `ollama pull` and pass to the API   'nomic-embed-text'
  alias        the registry key and table-name stem         'nomic_embed_text'

Eight notebooks referenced the alias ``all-minilm-l6-v2`` that no tag backed, and
two different notebooks normalized aliases differently, so the same model resolved
to two registry rows and two table names. Everything here exists to make that
class of bug unrepresentable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A legal unquoted PostgreSQL identifier: starts with a letter, then lowercase
# alphanumerics and underscores. Mirrored by a CHECK constraint on
# embedding_registry.model_alias, because convention alone already failed once.
ALIAS_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_SEPARATORS = re.compile(r"[.\-/:]+")
_RUNS = re.compile(r"_+")


def canonical_alias(name: str) -> str:
    """Normalize any model name into the single canonical alias form.

    This is the *only* function permitted to produce an alias. Never build one by
    hand, and never derive a table name with ``.replace(".", "_")`` -- that idiom
    is what produced two spellings of the same model.

    >>> canonical_alias('hf.co/CompendiumLabs/bge-base-en-v1.5-gguf')
    'hf_co_compendiumlabs_bge_base_en_v1_5_gguf'
    >>> canonical_alias('all-minilm-l6-v2') == canonical_alias('all_minilm_l6_v2')
    True
    >>> canonical_alias('nomic-embed-text')
    'nomic_embed_text'
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"model name must be a non-empty string, got {name!r}")

    alias = _SEPARATORS.sub("_", name.strip().lower())
    alias = _RUNS.sub("_", alias).strip("_")

    if not ALIAS_RE.match(alias):
        raise ValueError(
            f"{name!r} normalizes to {alias!r}, which is not a legal SQL identifier. "
            "Aliases must start with a letter and contain only [a-z0-9_]."
        )
    return alias


@dataclass(frozen=True)
class EmbeddingModel:
    """One embedding model, as both Ollama knows it and the registry knows it.

    ``alias`` is stored rather than derived from ``ollama_tag`` precisely because
    the two names are independent. Ollama publishes MiniLM-L6-v2 under the terse
    tag ``all-minilm``, but the curriculum has always called it
    ``all_minilm_l6_v2``, which says which MiniLM it is. Both names are correct
    for their own audience; the registry's job is to hold them together.
    """

    ollama_tag: str
    dimension: int
    note: str
    alias: str = ""
    normalized: bool = True

    def __post_init__(self) -> None:
        # Default the alias from the tag, but always canonicalize whichever we got.
        object.__setattr__(self, "alias", canonical_alias(self.alias or self.ollama_tag))


@dataclass(frozen=True)
class LanguageModel:
    ollama_tag: str
    note: str


# The catalog spans three dimensions deliberately. A single-dimension catalog lets
# a hardcoded 768 survive unnoticed -- which is exactly how `vector(768)` stayed
# in the DDL while the README advertised a 1024-dim model that could never work.
EMBEDDING_MODELS: dict[str, EmbeddingModel] = {
    m.alias: m
    for m in (
        EmbeddingModel(
            ollama_tag="all-minilm",
            alias="all_minilm_l6_v2",  # the name 8 notebooks already use
            dimension=384,
            note="Fastest and smallest. The MiniLM-L6-v2 the advanced tier always meant to use.",
        ),
        EmbeddingModel(
            ollama_tag="nomic-embed-text",
            dimension=768,
            note="Default. Strong quality per MB, CPU-friendly, 8192-token context.",
        ),
        EmbeddingModel(
            ollama_tag="mxbai-embed-large",
            dimension=1024,
            note="Highest quality of the three. Proves the schema is dimension-agnostic.",
        ),
    )
}

LANGUAGE_MODELS: dict[str, LanguageModel] = {
    "llama3.2:3b": LanguageModel("llama3.2:3b", "Default. ~2GB, comfortable on 8GB RAM."),
    "llama3.2:1b": LanguageModel("llama3.2:1b", "Fallback for constrained machines. Weaker synthesis."),
}

DEFAULT_EMBEDDING_ALIAS = "nomic_embed_text"
DEFAULT_LLM_TAG = "llama3.2:3b"


def get_embedding_model(alias_or_tag: str) -> EmbeddingModel:
    """Look up a catalog model by either of its names.

    Accepting both is deliberate: notebooks and docs refer to models by tag, while
    the database refers to them by alias, and callers should not have to care.
    """
    alias = canonical_alias(alias_or_tag)
    if alias in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[alias]

    # Fall back to matching the Ollama tag, so callers may use either name.
    for model in EMBEDDING_MODELS.values():
        if canonical_alias(model.ollama_tag) == alias:
            return model

    known = ", ".join(f"{a} ({m.ollama_tag})" for a, m in sorted(EMBEDDING_MODELS.items()))
    raise KeyError(
        f"{alias_or_tag!r} (alias {alias!r}) is not in the catalog. Known models: {known}. "
        "Add it to EMBEDDING_MODELS with its true dimension, then run scripts/preflight.py."
    ) from None
