---
dataset: tickets.json
count: 29
theme: "shipping delay"
ground_truth_count: 7
---

# Answer key -- ground truth for `tickets.json`

This file is the hand-computed answer key, same pattern as
`content/youtube/demo/2026-08-17-cortex-agents-mcp-server/sample_outputs.md`.
It is **not** read by the pipeline -- it exists so a human (or this report)
can check the pipeline's output against a known-correct number, the same way
Snowflake's own benchmark paper defines ground truth for its counting
questions.

## The question

> "How many tickets mention shipping delays?"

## Ground truth: **7 tickets**

| Ticket ID | Phrasing used (deliberately varied, no shared substring with "shipping delay") |
|---|---|
| T002 | "package is late" / tracking "hasn't moved" |
| T004 | "delivery is taking forever" |
| T007 | "shipment ... was delayed" / "stuck somewhere" |
| T010 | "still hasn't arrived" / 12 days, tracking "pending" |
| T013 | "box never showed up" / "lost somewhere" |
| T018 | "shipment is delayed in transit" |
| T029 | "sitting at the local carrier depot ... without moving" / "slow ... delivery" |

## Deliberate near-miss distractors (NOT counted -- shipping-adjacent but not a delay complaint)

These exist specifically to test whether retrieval/generation over-generalizes
on the keyword "shipping" instead of the theme "shipping delay":

| Ticket ID | Why it's excluded |
|---|---|
| T006 | Shipping **address change request** before the item ships -- not a complaint about lateness |
| T011 | **Wrong item/color** shipped -- a fulfillment accuracy issue, not a delay |
| T017 | Item **arrived on time but damaged** -- explicitly says "showed up on time" |

## All other tickets (19)

Billing (T001, T020, T026), account/login (T003, T023, T024), subscription
management (T008, T012), product questions/defects (T009, T025), feature
requests (T005, T022), praise (T015, T027), app bugs (T016, T028), other
(T014, T019, T021) -- none mention shipping timing at all.

## How to grade a pipeline's answer

- **Exact match on the count (7)** is the pass/fail bar for this test, per
  Snowflake's "Beyond RAG" framing: corpus-wide counting/aggregation is what
  standard top-k RAG is bad at.
- Also worth checking **which ticket IDs** a retrieval step actually
  surfaced vs. this table -- a pipeline can get the number 7 by luck while
  missing/adding tickets, or (more likely, per the published finding) return
  a number below 7 because top-k only ever surfaces a handful of the 7 true
  matches out of the full 29-ticket corpus.
