# Mechanism → Syndrome Composition Map

Working reference for the composable-mechanism-LoRA design (see `idea.md`). Goal: a defensible
set of *atomic* cognitive/affective mechanisms, each human-validated and clinically anchored, plus
the **composition recipes** that combine them into syndromes.

Confidence tags: **[strong]** meta-analytic/longitudinal human evidence · **[moderate]** validated
construct + instrument, evidence concentrated in anxiety/depression · **[framework]** organizing
taxonomy, not a single isolated mechanism.

---

## 1. The atomic mechanism set

Backbone = **Harvey, Watkins, Mansell & Shafran (2004)** — 14 transdiagnostic processes in 5
categories. This is the canonical "atomic" taxonomy and maps cleanly onto separate LoRAs. Each is
documented as transdiagnostic (maintains symptoms across multiple disorders), which is exactly the
property we want for composability.

| Category | Atomic mechanism | Validated instrument | Conf. |
|---|---|---|---|
| **Attention** | Threat-vigilance, external (attentional bias to threat) | Dot-probe (Bar-Haim et al. 2007 meta-analysis) | [strong] |
| | Self-focused attention, internal | Self-focus / private self-consciousness scales | [moderate] |
| | Attentional avoidance | Dot-probe (avoidance pattern) | [moderate] |
| **Memory** | Recurrent / intrusive memories | IES-R; PTSD re-experiencing items | [strong] |
| | Overgeneral memory | Autobiographical Memory Test (AMT) | [moderate] |
| | Mood-congruent selective memory | Recall-bias paradigms | [moderate] |
| **Reasoning** | Interpretation bias (threat/negative) | Ambiguous-scenario tasks; interpretation bias paradigms | [moderate] |
| | Expectancy bias | — | [moderate] |
| | Emotional reasoning ("I feel it, so it's true") | — | [moderate] |
| **Thought** | **Repetitive negative thinking (RNT)** = rumination + worry | Perseverative Thinking Questionnaire (PTQ); Ruminative Response Scale; PSWQ | **[strong]** |
| | Metacognitive beliefs (pos/neg about thinking) | MCQ-30 | [moderate] |
| | Thought suppression | White Bear Suppression Inventory | [moderate] |
| **Behaviour** | **Experiential / behavioural avoidance** | AAQ-II (note: construct-validity debate); BADS | **[strong]** |
| | Safety behaviours | Disorder-specific checklists | [moderate] |

**Add-ons not in Harvey but strongly evidenced as atomic units:**

| Mechanism | Validated instrument | Conf. | Note |
|---|---|---|---|
| **Emotion dysregulation / lability** | DERS; Linehan biosocial model | **[strong]** | Core of BPD; bridges to depression & PTSD |
| **Intolerance of uncertainty (IU)** | IUS-12 | **[strong]** | Core of GAD; transdiagnostic via emotion-dysreg + rumination mediation |
| Negative self-schema / cognitive triad | Dysfunctional Attitudes Scale (DAS); ATQ | [moderate] | Beck's model — self/world/future; *catastrophizing* sits here as an interpretation+RNT compound |
| Dichotomous ("splitting") thinking | — (weak standalone instrument) | [weak] | Often treated as a *style* on top of emotion dysregulation in BPD, not a clean atomic unit |

> **Design note:** "catastrophizing" and "learned helplessness" are *compounds*, not atoms —
> catastrophizing ≈ threat interpretation bias + RNT; helplessness ≈ negative expectancy bias +
> behavioural avoidance. Train the atoms; let the compound emerge from the merge. This is a testable
> mechinterp prediction.

---

## 2. Syndrome composition recipes (which atoms compose which disorder)

Anchored to Beck (depression), Borkovec/Dugas (GAD), Ehlers-Clark (PTSD), Linehan (BPD).

### Depression (Beck cognitive model + RNT)
- **Rumination** (RNT, brooding subtype) — core maintaining process **[strong]**
- **Negative self-schema / negative automatic thoughts** (cognitive triad: self, world, future)
- Overgeneral autobiographical memory
- Mood-congruent memory bias
- Self-focused attention
- Behavioural avoidance / withdrawal
- (historical: learned helplessness → expectancy bias + avoidance)

### Generalized Anxiety Disorder (GAD)
- **Worry** (RNT, future-oriented subtype) — core
- **Intolerance of uncertainty** — key distinguishing atom **[strong]**
- Threat interpretation bias
- Positive *and* negative metacognitive beliefs about worry
- Avoidance / safety behaviours

### PTSD
- **Intrusive / recurrent memories** — core re-experiencing
- **Hypervigilance** = attentional threat bias **[strong, dot-probe]**
- **Avoidance** (experiential + behavioural) — core cluster
- Negative cognitions about self/world (negative self-schema variant)
- Thought suppression (paradoxically *maintains* intrusions — rebound effect)
- Rumination (post-trauma)

### Borderline Personality Disorder (BPD) — Linehan biosocial, longitudinally validated
Validated in a 3-wave longitudinal study (332 adolescents; Cambridge *Dev & Psychopathology*):
- **Emotional vulnerability → emotion dysregulation** (b=0.19, p=.013) **[strong]**
- **Impulsivity** → emotion dysregulation (β=.20) and → BPD symptoms (β=.17) **[strong]**
- **Invalidating environment** → emotion dysregulation (β=.10) *(developmental input, not a runtime cognitive atom)*
- **Emotion dysregulation ↔ BPD symptoms** bidirectional feedback loop (β=.28 / β=.13) **[strong]**
- Dichotomous/splitting thinking + identity disturbance ride on top of the dysregulation core

---

## 3. Comorbidity / bridge structure (why merges should interact, not just add)

This is the part most relevant to the mechinterp "do circuits superpose / produce emergent
comorbidity?" question.

- **RNT is the core shared node between depression and anxiety** — commonality analysis: ~16% of RNT
  variance is shared rumination–worry; largest single component is a 9.8% four-way overlap
  (rumination/worry/depression/anxiety). **[strong]** → predicts a rumination+worry LoRA merge should
  reproduce dep–anx comorbidity.
- **Emotion dysregulation bridges depression, PTSD, and BPD.** **[strong]**
- **Intolerance of uncertainty bridges GAD ↔ depression ↔ PTSD**, mostly *indirectly* via emotion
  dysregulation and rumination (mediation, not direct). **[moderate]**
- **Network/bridge-symptom findings** (for behavioral grounding of comorbidity, symptom level not
  mechanism level): psychomotor agitation/retardation = strongest depression↔anxiety bridge; sleep
  disruption = key PTSD↔depression bridge; concentration problems also bridge.

> **Structural backbone caution (from verified round):** do **not** use a single "p-factor" as a
> mechanistic unit — the strong p-factor claim was refuted (0-3) and p "means hundreds of things"
> depending on indicators. Use a **three-dimension hierarchy** (internalizing / externalizing /
> thought-disorder, à la HiTOP) or **RDoC domains** as the composition scaffold instead.

---

## 4. RDoC dimensional anchor (for affective atoms)

RDoC **Negative Valence Systems** give 5 self-report-measurable affective dimensions usable as
affective atoms: Acute Threat/Fear, Potential Threat/Anxiety, Sustained Threat, Loss, Frustrative
Nonreward (Brake et al. 2022, N=1,509 factor analysis recovers them). **[strong]** Caveat: Potential
Threat and Loss have weak discriminant validity (proposed merge into "Distress") — bounds how
orthogonal anxiety vs. depression affective atoms can be.

---

## 5. Open gaps (not yet source-verified to the same bar)

- Standardized **language/behavior coding schemes** per mechanism (needed for RL reward functions &
  SFT quality filters) — only partially found. Candidates: AMT for overgeneral memory, dot-probe for
  threat bias, LIWC-style markers for rumination — need dedicated sourcing.
- Clean instrument for **dichotomous/splitting thinking** as a standalone atom.
- Per-mechanism deep validation for: emotional reasoning, expectancy bias, metacognitive beliefs.

---

## Key sources
- Harvey, Watkins, Mansell & Shafran (2004), *Cognitive Behavioural Processes Across Psychological Disorders*, OUP — https://academic.oup.com/book/1289
- Wade, Pennesi, Pellizzer (2025), 38-process transdiagnostic expansion — https://journals.sagepub.com/doi/10.1177/00048674241312803
- Biosocial BPD longitudinal validation, *Dev & Psychopathology* — https://www.cambridge.org/core/journals/development-and-psychopathology/article/validating-the-biosocial-model-of-borderline-personality-disorder-findings-from-a-longitudinal-study/D8B0BA1BD36F602A72C5101490CF7AE7
- RNT commonality analysis — https://pmc.ncbi.nlm.nih.gov/articles/PMC6370308/
- RNT CBT meta-analysis (55 RCTs) — https://pmc.ncbi.nlm.nih.gov/articles/PMC12017360/
- Experiential avoidance meta-analysis (441 studies) — https://www.sciencedirect.com/science/article/abs/pii/S221214472200028X
- Attentional threat bias meta-analysis (Bar-Haim et al. 2007) — https://people.socsci.tau.ac.il/mu/dominiquelamy/files/2014/08/Bar-Haim_et_2007.pdf
- RDoC NVS factor structure (Brake et al. 2022) — https://pubmed.ncbi.nlm.nih.gov/36229109/
- Depression–anxiety bridge-symptom comorbidity — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7523307/
