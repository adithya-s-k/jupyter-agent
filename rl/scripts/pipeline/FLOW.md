# Pipeline flow — visualized

A picture-first reference for the staged-verification pipeline. Every diagram
below describes one slice of the same system: how a single task gets from
"manifest row" to "verified, with difficulty label" (or to "dropped, with
reason"). Read top-to-bottom for the full mental model, or jump to whichever
diagram answers your question.

All diagrams are Mermaid — they render in GitHub, on Hugging Face dataset
cards, and in any modern Markdown previewer.

---

## 1. One task, end to end

The big picture. Solid arrows are the happy path; dashed are doctor-driven
recovery branches.

```mermaid
flowchart TD
    M([manifest row<br/>task_id · question · gold<br/>kaggle · reward_mode]) --> A

    A[<b>Phase A — build_spec</b><br/>generate scaffold<br/>task.toml + instruction.md<br/>environment/ + tests/]
    A --> SB[(<b>pending/</b><br/>scaffold lives here)]

    SB --> B[<b>Phase B — anchor trials</b><br/>Sonnet + seta agent<br/>up to k_max retries<br/>inside Docker container]
    B --> BR{reward<br/>== 1.0 ?}

    BR -->|yes| D
    BR -->|"no (doctor disabled)"| PBF[(<b>phase_b_failed/</b><br/>Stage-1 cohort terminal)]
    BR -->|"no (doctor enabled)"| C

    C[<b>Phase C — doctor</b><br/>Sonnet tool-loop<br/>read · probe · edit · finalize]
    C --> CV{finalize<br/>verdict}

    CV -. spec_fixed / .-> B2
    CV -. gold_corrected .-> B2
    CV -- verifiable_judge --> D
    CV -- unverifiable --> DR[(<b>dropped/</b>)]

    B2[<b>Phase B2</b><br/>re-run anchor with<br/>edited spec or gold]
    B2 --> B2R{reward<br/>== 1.0 ?}
    B2R -- yes --> D
    B2R -- no --> DR

    D[<b>Phase D — categorize</b><br/>Sonnet rubric<br/>→ difficulty_level 1-5]
    D --> VER[(<b>verified/</b><br/>with L1-L5 label)]

    style A fill:#1e3a5f,stroke:#59a6ff,color:#fff
    style B fill:#1e3a5f,stroke:#59a6ff,color:#fff
    style C fill:#3a2e1e,stroke:#e6b455,color:#fff
    style B2 fill:#1e3a5f,stroke:#59a6ff,color:#fff
    style D fill:#1e3a3a,stroke:#3ec97a,color:#fff
    style VER fill:#1e3a1e,stroke:#3ec97a,color:#fff
    style DR fill:#3a1e1e,stroke:#e25c5c,color:#fff
    style PBF fill:#3a3a1e,stroke:#e6b455,color:#fff
```

---

## 2. Inside Phase B — the retry + regrade loop

`--k-max` retries on flaky LLM behaviour. After each fail, the regrade fallback
re-grades the SAME trajectory under a looser `reward_mode` (e.g. `numeric` →
`flexible` → `llm_judge`) — no new LLM call to the agent, just a cheaper grader.

```mermaid
flowchart TD
    start([Phase B start])
    start --> k1[k=1<br/>spawn trial<br/>via harbor run --env docker]
    k1 --> graded{grader.py:<br/>reward == 1.0?}

    graded -- yes --> PASS([Phase B pass<br/>reward=1.0])
    graded -- no --> regrade[<b>regrade fallback</b><br/>re-grade trajectory<br/>under looser mode]
    regrade --> regraded{passes under<br/>looser mode?}
    regraded -- yes --> promote[<b>promote</b> trial<br/>+ update reward_mode_initial<br/>in task.toml]
    promote --> PASS
    regraded -- no --> kcheck{k < k_max?}
    kcheck -- yes --> kinc[k += 1]
    kinc --> k1
    kcheck -- no --> FAIL([Phase B fail<br/>→ doctor OR<br/>phase_b_failed])

    style PASS fill:#1e3a1e,stroke:#3ec97a,color:#fff
    style FAIL fill:#3a1e1e,stroke:#e25c5c,color:#fff
    style promote fill:#1e3a3a,stroke:#3ec97a,color:#fff
```

**Key knobs**
- `--k-max 2` (default) — first attempt + one retry
- `--max-retries-per-trial 1` — per-trial retry on **transient** errors only (SandboxException, 5xx, TimeoutException). Distinct from k-max (which retries on legitimate failure).
- `--regrade-judge-model openai/gpt-5.4-nano` — the cheap LLM judge used by the regrade fallback

---

## 3. Inside Phase C — the doctor

Sonnet (the "doctor brain") gets the failing trajectory, the spec, and the gold,
then emits tool calls **one at a time**. It can run mini-Phase-B trials with
other models to cross-check, edit the spec, or declare the task unverifiable.
A session is bounded by tool-call count, $ budget, and number of allowed rewrites.

```mermaid
flowchart TD
    DS([doctor_start])
    DS --> sys[system prompt:<br/>failing trajectory + spec + gold]
    sys --> turn[<b>doctor_turn</b><br/>Sonnet decides next action]
    turn --> tool{tool call}

    tool -- read_file --> t1[inspect any file<br/>in the scaffold]
    tool -- preview_dataset --> t2[sample the<br/>Kaggle CSV]
    tool -- list_files --> t3[ls /home/user/input/<br/>inside container]
    tool -- read_trajectory --> t4[read failing<br/>trial trajectory]
    tool -- probe_with_model --> t5[<b>spawn mini Phase-B trial</b><br/>nano · opus · gpt-5.5 · gpt-5.5-codex<br/>· qwen · glm · kimi · deepseek]
    tool -- edit_task_toml --> t6[rewrite gold_answer,<br/>reward_mode, ATOL/RTOL...]
    tool -- edit_instruction --> t7[rewrite instruction.md]
    tool -- finalize --> done

    t1 --> turn
    t2 --> turn
    t3 --> turn
    t4 --> turn
    t5 --> turn
    t6 --> turn
    t7 --> turn

    done{<b>finalize</b><br/>verdict}
    done -- spec_fixed --> B2[trigger Phase B2<br/>with edited spec]
    done -- gold_corrected --> B2g[trigger Phase B2<br/>with new gold]
    done -- verifiable_judge --> JJ[verified via judge<br/>no B2 needed]
    done -- unverifiable --> DROP[final: dropped]

    note[<b>Bounded by</b><br/>doctor_max_calls = 20<br/>doctor_budget = $0.50<br/>max_rewrites = 1<br/>probe_timeout_sec = 180s]
    note -.-> turn

    style DS fill:#3a2e1e,stroke:#e6b455,color:#fff
    style turn fill:#3a2e1e,stroke:#e6b455,color:#fff
    style t5 fill:#1e3a5f,stroke:#59a6ff,color:#fff
    style B2 fill:#1e3a5f,stroke:#59a6ff,color:#fff
    style B2g fill:#1e3a5f,stroke:#59a6ff,color:#fff
    style JJ fill:#1e3a1e,stroke:#3ec97a,color:#fff
    style DROP fill:#3a1e1e,stroke:#e25c5c,color:#fff
```

**The four finalize verdicts and what they mean**

| Verdict | Doctor's reasoning | Next step |
|---|---|---|
| `spec_fixed` | grader was too strict; e.g. `numeric` should be `flexible` | Phase B2 with the edited `task.toml` |
| `gold_corrected` | multiple probes converged on a NEW answer ≠ original gold | Phase B2 with the new `gold_answer` |
| `verifiable_judge` | LLM judge confirms anchor's predicted answer ≡ gold semantically | skip B2; final verdict = `verifiable_judge` |
| `unverifiable` | probes give inconsistent answers; question is genuinely ambiguous | final verdict = `dropped` |

---

## 4. Scaffold state machine

Each scaffold lives in exactly one folder. The folder *is* the state — by
inspecting `harbor/tasks/<suite>/{pending,verified,dropped,phase_b_failed}/`,
you can see the current verdict for every task without parsing JSONL.

```mermaid
stateDiagram-v2
    direction LR

    [*] --> pending : Phase A · build_spec()

    pending --> verified : Phase B pass<br/>(verified)
    pending --> verified : Phase B2 pass<br/>(verified_after_rewrite)
    pending --> verified : Phase B2 pass<br/>(verified_gold_corrected)
    pending --> verified : doctor·verifiable_judge
    pending --> phase_b_failed : Phase B fail<br/>(--skip-doctor)
    pending --> dropped : doctor·unverifiable

    phase_b_failed --> verified : Stage 2 recovery
    phase_b_failed --> dropped : Stage 2 doctor drop
    phase_b_failed --> phase_b_failed : Stage 2 still fails

    verified --> [*]
    dropped --> [*]
```

`build.py:_resolve_task_dir()` searches all four folders, so re-runs find the
existing scaffold instead of rebuilding. The folder moves are atomic
(`shutil.move()`) at the end of `process_task()`.

---

## 5. Two-stage workflow on a batch

How you actually use the pipeline in practice. Stage 1 processes the whole
batch cheaply; Stage 2 picks up only the failures from Stage 1.

```mermaid
flowchart LR
    P([candidate task IDs<br/>e.g. 2000 train tasks])

    P --> S1[<b>Stage 1</b><br/>Sonnet anchor<br/>categorize on pass<br/>no doctor]

    S1 --> V1["verified/<br/>40-50 pct<br/>with L1-L5"]
    S1 --> F1["phase_b_failed/<br/>50-60 pct"]

    F1 -.->|extract ids| IDS([cache/stage2_ids.json])
    IDS --> S2[<b>Stage 2</b><br/>Doctor enabled<br/>categorize on recovery<br/>lean probe roster]

    S2 --> V2["verified/<br/>+25-35 pct<br/>recovered"]
    S2 --> D2["dropped/<br/>20-25 pct<br/>unverifiable"]
    S2 --> F2["phase_b_failed/<br/>5 pct<br/>unrecoverable"]

    style S1 fill:#1e3a5f,stroke:#59a6ff,color:#fff
    style S2 fill:#3a2e1e,stroke:#e6b455,color:#fff
    style V1 fill:#1e3a1e,stroke:#3ec97a,color:#fff
    style V2 fill:#1e3a1e,stroke:#3ec97a,color:#fff
    style F1 fill:#3a3a1e,stroke:#e6b455,color:#fff
    style F2 fill:#3a3a1e,stroke:#e6b455,color:#fff
    style D2 fill:#3a1e1e,stroke:#e25c5c,color:#fff
```

**Why two stages.** Stage 1 is cheap-and-fast — most tasks Sonnet handles cleanly
go straight to "verified+categorized" without paying for the doctor. Stage 2 is
the targeted-rescue stage — it pays doctor + probe costs only on the residual.
On the 500-task eval the split was:

| Stage | Output | Cost |
|---|---|---|
| Stage 1 | ~50% verified (with difficulty), ~50% phase_b_failed | $0.08 / task |
| Stage 2 | ~60% recovered (verified-class), ~40% dropped or stuck | $0.20 / task in this stage |
| **Combined** | **73% verified end-to-end** | **~$0.20 / verified-task** |

---

## 6. What lands in `state.jsonl` (sequence view)

Every meaningful event during `process_task()` gets one JSONL line. State.jsonl
is the source of truth for live monitoring (the textual stdout log can be
block-buffered; state.jsonl is fsync'd more aggressively).

```mermaid
sequenceDiagram
    participant CLI as cli.py
    participant Orch as orchestrator
    participant H as harbor (subprocess)
    participant State as state.jsonl

    CLI->>Orch: process_task(row)
    Orch->>State: task_start

    Note over Orch: Phase A
    Orch->>Orch: build_spec()
    Orch->>State: spec_built

    Note over Orch,H: Phase B  (loop k=1..k_max)
    loop until pass or k_max
        Orch->>State: trial_start
        Orch->>H: run_trial(harbor run --env docker)
        H-->>Orch: TrialResult (reward, predicted, cost)
        Orch->>State: trial_finish

        alt reward < 1.0
            Orch->>Orch: regrade fallback
            opt regrade promotes
                Orch->>State: regrade_promoted
            end
        end
    end

    alt Phase B failed AND doctor enabled
        Note over Orch,H: Phase C  (doctor tool-loop)
        Orch->>State: doctor_start
        loop until finalize
            Orch->>State: doctor_turn
            Orch->>State: doctor_tool
            opt probe_with_model
                Orch->>State: probe_start
                Orch->>H: run probe trial
                H-->>Orch: probe TrialResult
                Orch->>State: probe_finish
            end
        end
        Orch->>State: doctor_finish

        opt doctor: spec_fixed or gold_corrected
            Note over Orch,H: Phase B2  (re-run with rewrite)
            Orch->>State: trial_start (rewrite_idx=1)
            Orch->>H: run_trial(edited spec)
            H-->>Orch: TrialResult
            Orch->>State: trial_finish
        end
    end

    opt verdict ∈ verified-class
        Note over Orch: Phase D
        Orch->>State: categorize_finish (level, confidence)
    end

    Orch->>State: task_finish (verdict, total_cost)
    Orch->>CLI: done
```

You can replay/inspect a run by grepping these events:

```bash
# Did this task ever pass?
grep '"task_id": "0000/419/419825.ipynb_qa_1"' runs/*/state.jsonl | grep '"reward": 1'

# What did the doctor decide?
grep '"event": "doctor_finish"' runs/*/state.jsonl | head

# Per-task wall time
python -c "
import json, glob
events = [json.loads(l) for l in open('runs/<ts>/state.jsonl')]
starts = {e['task_id']: e['ts'] for e in events if e['event']=='task_start'}
ends = {e['task_id']: e['ts'] for e in events if e['event']=='task_finish'}
for tid in starts:
    if tid in ends:
        print(tid, ends[tid], '-', starts[tid])
"
```

---

## 7. Where the dollars go

Real numbers from the 500-task eval (combined Stage 1 + Stage 2 + categorize-backfill):

```mermaid
flowchart LR
    SPEND(["Total $71.93 — $0.20 per verified-task"]) --> B["<b>Phase B</b><br/>anchor seta-loop<br/>$48.82 — 68 pct"]
    SPEND --> P["<b>Doctor probes</b><br/>nano + gpt-5.5 + others<br/>$13.86 — 19 pct"]
    SPEND --> DR["<b>Doctor brain</b><br/>Sonnet diagnose loop<br/>$7.56 — 11 pct"]
    SPEND --> C["<b>Phase D categorize</b><br/>Sonnet rubric<br/>$1.83 — 3 pct"]

    style B fill:#1e3a5f,stroke:#59a6ff,color:#fff
    style P fill:#3a2e1e,stroke:#e6b455,color:#fff
    style DR fill:#3a2e1e,stroke:#e6b455,color:#fff
    style C fill:#1e3a1e,stroke:#3ec97a,color:#fff
```

### Average per-task costs (Sonnet anchor + Sonnet doctor)

| What you're paying for | Avg $ | When it fires |
|---|---:|---|
| Phase B trial (one anchor attempt) | **~$0.05–0.10** | Every task in Stage 1; every Stage 2 task does it too |
| Phase B regrade fallback | **~$0.001** | After each failing trial; just an LLM-judge call |
| Phase B2 re-run (after rewrite) | **~$0.05–0.10** | ~10% of Stage 2 tasks (when doctor edits spec or gold) |
| Doctor session (brain + tool calls) | **~$0.02–0.05** | When Phase B fails AND Stage 2 is enabled |
| Doctor probe (one mini-trial via nano/gpt-5.5) | **~$0.03–0.10** | 1-3 probes per doctor session typically |
| Phase D categorize (single rubric call) | **~$0.005** | Every verified-class task |
| **End-to-end avg per task** | **~$0.10–0.15** | Stage 1 alone: ~$0.08 · Stage 2 alone: ~$0.20 |
| **Per verified-class task** | **~$0.20** | $ spent ÷ tasks that ended verified |

### Small concrete example: the 500-task eval run

The full eval — 500 tasks, Stage 1 + Stage 2 on the failures, plus a
post-run L0 backfill — looked like this:

| Phase | Tasks processed | $ spent | Avg $/task | Wall time |
|---|---:|---:|---:|---:|
| Stage 1 on 100 (batch 1) | 100 | $10.84 | $0.11 | 11 min |
| Stage 2 on 51 phase_b_failed | 51 | $10.15 | $0.20 | 23 min |
| Stage 1 on 278 (batch 2: 100 fresh + 173 new + 5 old) | 278 | ~$15 | $0.05 | 37 min |
| Stage 2 on 168 phase_b_failed | 168 | $22.77 | $0.14 | 27 min |
| L0 backfill (re-categorize 27 missing) | 27 | $0.13 | $0.005 | <1 min |
| Categorize-gate fix (catch verifiable_judge in-pipeline) | — | $0.13 | — | — |
| **Total** | **500** (one-shot) | **$71.93** | **$0.14 avg / $0.20 per verified** | **~1.5 hours active** |

End state: **366 verified** (73%), **127 dropped** (25%), **7 phase_b_failed**
residue (1%). All 366 carry an L1-L5 difficulty label.

If you want a quick sanity check that everything's wired right, **a single
L1 task end-to-end costs ~$0.02 with the bash agent + Sonnet** (one Phase B
trial, no doctor, one categorize call). That's the smallest meaningful unit
the pipeline produces.

**Implication.** Phase B is the dominant cost line at ~⅔ of the total.
Making it faster or cheaper (via a cheaper anchor model, smarter k-max,
or better regrade heuristics) moves the needle more than optimising the
doctor. The doctor's cost is split ~64% probes / ~36% brain — and probes ARE
mostly Phase-B-equivalents under different models. So really the "cost of
running an LLM-tool-loop on a Kaggle task" is what we're paying for.

---

## 8. Reading the diagrams in three lines

- **Phase A** = generate scaffold, idempotent.
- **Phase B** = anchor LLM tries the task in Docker, with retries + regrade fallback. If pass → Phase D. If fail → Phase C (or `phase_b_failed/` if doctor disabled).
- **Phase C** = doctor LLM with 8 tools. Can probe other models, edit the spec/gold, declare unverifiable, or judge-confirm. Outcomes route to Phase B2 (re-run) or directly to a final bucket.
- **Phase D** = single rubric call assigns L1-L5 difficulty on verified-class tasks.
