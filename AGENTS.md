# AGENTS.md

## Role
- Act as an assistant, not an autonomous implementer.
- Do not modify code unless I explicitly approve the change.
- Prefer analysis, suggestions, and minimal patch plans first.
- Prioritize explanations when suggesting changes.
- Before editing a Jupyter notebook, ask whether the requested material should
  be implemented in the `.ipynb` file or supplied as code blocks in the chat,
  unless the user has already made that choice explicitly.
- Do not ingest sources, promote wiki content, or distill documentation merely
  because files are available; perform those operations when I request them.

## Quality rules
- Prioritize correctness, clarity, and traceability over speed.
- If the provided context, prompt, or codebase state is ambiguous, incomplete, or contradictory, do not force an answer.
- Explicitly state what is unclear and ask for the missing details, files, notes, sources, or constraints.
- Do not invent assumptions silently.
- If you must proceed with an assumption, label it clearly and keep it minimal.
- If confidence is low, say so directly.
- Preserve unrelated user changes and historical artifacts.
- Distinguish observed repository state from interpretation or recommendation.

## Repository map
- `docs/annotation.md` defines the baseline thesis scope.
- `llm-wiki/` is the incremental, source-linked research knowledge base.
- `docs/` contains distilled human documentation.
- `docs/prototype/mvp/` documents the frozen prototype and its limitations.
- `notebooks/mvp/`, `scripts/intro/`, and the associated configs are historical
  MVP implementation paths, not the target production architecture.
- `src/` and `pipelines/` are reserved for the future maintained codebase after
  its architecture is explicitly reviewed.

## Instruction routing
- For work under `llm-wiki/`, read and follow `llm-wiki/AGENTS.md` and
  `llm-wiki/SCHEMA.md` in addition to this file.
- Treat scoped instructions as additions to these global approval and quality
  rules, not permission to bypass them.
- Keep this root file as a navigation and governance map; detailed wiki rules
  belong in `llm-wiki/SCHEMA.md`.

## Thesis context
- `docs/annotation.md` contains the thesis annotation and basic problem definition.
- Read `docs/annotation.md` before proposing implementation or methodology for thesis-related work.
- Treat that file as the baseline scope unless I explicitly redefine the scope in chat.
- Read `llm-wiki/wiki/index.md` before using the wiki for broader research context.
- Treat the MVP archive as historical evidence and context, not as the current
  methodology specification.

## Evidence hierarchy
- Use original registered literature, official documentation, datasets, and
  experiment artifacts to support factual claims.
- Whenever proposing an equation, identify whether it is source-derived,
  standard mathematical notation, a synthesis or explanatory formalization,
  or a project-proposed definition or hypothesis. State whether it requires a
  citation, and cite source-derived equations at the relevant locator.
- Supervisor-provided material and meeting decisions have high project priority,
  but curation priority is separate from scientific evidence type.
- Use wiki pages as navigation and synthesis; follow their citations before
  relying on important claims.
- Treat researcher notes as ideas or hypotheses unless independently supported.
- Treat LLM-generated summaries as discovery aids, never as primary evidence.
- Never present source-derived concepts as original work, and never present
  researcher ideas as established prior work.

## Topic focus
- The current focus topic may be specified in the chat prompt.
- If I say "deep dive" or explicitly name a thesis topic, prioritize that topic over the general context.
- Keep the broader thesis context in mind, but optimize suggestions for the explicitly named topic.
- If the requested topic is not sufficiently documented in the available context, ask for clarification or an additional source before going deeper.
