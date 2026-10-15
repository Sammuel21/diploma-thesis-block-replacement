---
title: Grilling Protocol
status: draft
modified: 2026-10-10
---

# Grilling Protocol

This protocol is adapted from the `grilling` skill by Matt Pocock in
[mattpocock/skills](https://github.com/mattpocock/skills/blob/95249b0b49782349740fd9b8c6ce32b4e59e497a/skills/productivity/grilling/SKILL.md),
commit `95249b0`, used under the MIT License reproduced at the end of this
document. The design-tree rounds, numbered questions with recommended
answers, and the split between facts the agent finds and decisions the user
makes come from the original. The project changes are listed in
[Changes from the original](#changes-from-the-original).

## Purpose

Grilling reduces miscommunication between the researcher and AI agents. The
agent asks structured questions until both sides share one understanding of a
plan, before any work starts.

## When to use

Use it only when the researcher asks for it, for example "grill me",
`/grilling`, or "stress-test this plan". Typical cases are designing a new
experiment class, preparing a plan for an implementation agent, or anything
that will spend GPU time. Do not start it for routine tasks.

## Procedure

1. Map the plan as decisions. A decision can be asked once the decisions it
   depends on are settled.
2. Ask the askable decisions in one numbered round, highest impact first, at
   most seven per round. Give each question a recommended answer with a
   one-line reason. A question that depends on another open question in the
   same round waits for a later round.
3. Look up facts yourself in the repository, result artifacts, and
   documentation. Never ask the researcher for something you can read. Do not
   spawn sub-agents unless the researcher asks.
4. For operational decisions, "yes" accepts the recommendation. For scientific
   decisions, such as controls, comparisons, budgets, and claims, the
   researcher answers with a short reason rather than only "yes".
5. After each round, recompute which decisions are askable. Stop when no open
   decisions remain and nothing is silently assumed.
6. Finish with a design summary: settled decisions, remaining risks, and
   proposed entries for the [shared decision log](decisions.md). The
   researcher may edit or strike entries. Confirming the summary approves
   writing the remaining entries to the log; write them then, without a
   separate request. Do not edit other files or implement anything until the
   researcher gives a separate go.

## Question format

```markdown
❓ **Q1 — <title>**: <question, with options when useful>

➡️ Recommended: <answer> — <one-line reason>

---

❓ **Q2 — <title>**: <question>

➡️ Recommended: <answer> — <one-line reason>
```

## Changes from the original

- Rounds hold at most seven questions instead of the whole open frontier, to
  limit fatigue on large research designs.
- The agent looks up facts itself instead of dispatching sub-agents, following
  the repository's delegation rules in `AGENTS.md`.
- Scientific decisions require a short researcher reason, so a recommended
  answer is not accepted by default.
- The session ends with proposed entries for the shared decision log, so
  settled decisions remain visible to every agent.
- The companion `domain-modeling` skill, used by `grill-with-docs`, is not
  adopted. Its root glossary and ADR folder would duplicate the wiki decision
  and concept pages and write files without approval.

## Original license

```text
MIT License

Copyright (c) 2026 Matt Pocock

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
