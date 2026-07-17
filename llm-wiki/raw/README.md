# Raw Sources

`raw/` is the source boundary of the LLM wiki.

Source binaries do not all live inside this Git repository. Large PDFs and
mutable idea folders remain in the surrounding diploma-thesis workspace and
are registered in `collections.yml`. This avoids duplicating large files while
making their role and authority explicit.

## Files

- `collections.yml` registers source collections and their default treatment.
- `sources.yml` registers individual sources once they enter the controlled
  ingestion workflow.
- `legacy/` contains repository-tracked snapshots of earlier notes and source
  lists. These files are migration input, not canonical knowledge.

## Immutability rule

Agents may read raw material but must not rewrite it. A mutable external note
can still be registered, but ingestion must capture its path, date, and SHA-256
hash so later changes cannot silently alter the evidence base. Corrections are
new source versions or explicit annotations in the wiki.

Collection registration does not imply that every item has been read,
validated, or accepted as evidence.
