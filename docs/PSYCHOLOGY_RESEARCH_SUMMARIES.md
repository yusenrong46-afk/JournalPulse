# Psychology and cognitive-science evidence for Luna

Accessed October 5, 2026. This report supersedes the earlier memory-based
synthesis. Browser automation succeeded after repairing Chromium's access to its
existing NSS certificate store; HTTPS verification stayed enabled.

## Reading scope

| Source | What was actually retrieved and read |
| --- | --- |
| NICE NG222 | Relevant recommendation sections 1.3, 1.4, 1.5 and Table 1; not every clinical recommendation or supporting evidence review |
| Uphoff et al. (2020) | Cochrane's official plain-language summary and structured abstract; full review inaccessible |
| Braun et al. (2015) | Full author-manuscript article via NCBI PMC XML: introduction, methods, results, discussion, and scale appendix |
| Lieberman et al. (2007) | Indexed abstract from Europe PMC; publisher full text blocked |
| Webb et al. (2012) | Indexed abstract from Europe PMC; publisher full text blocked |
| Bonanno & Burton (2013) | Indexed abstract from Europe PMC; publisher full text blocked |
| Ryan & Deci (2000) | Indexed abstract from Europe PMC; author/publisher full text inaccessible |

Europe PMC records reproduce the indexed abstracts, not independent replications.
Short quotations below were matched exactly against retrieved text. Abstract-only
summaries cannot substitute for assessment of complete methods and supplementary
analyses. This is a targeted review of the requested references, not a systematic
or exhaustive review of contemporary literature.

## 1. NICE: collaborative choice and structured support

**Source:** [Depression in adults: treatment and management, NG222](https://www.nice.org.uk/guidance/ng222/chapter/Recommendations).

**Evidence:** A clinical guideline rather than an experiment. Recommendation
1.3.1 asks about contributing factors, prior helpful treatments, preferences, and
desired outcomes. Section 1.3.3 includes declining treatment or changing one's
mind. Table 1 describes structured guided self-help with trained-practitioner
support, typically over 6–8 sessions, and distinguishes CBT from behavioural
activation.

**Verified excerpt, 1.3.1:** “what they hope to gain from treatment”

**Interpretation for Luna:** Start with the user's goal, preserve the ability to
decline suggestions, and distinguish exploratory conversation from action
planning. Structure should organize the interaction without imposing a fixed
sequence regardless of user intent.

**Boundary:** These recommendations concern depression treatment with specified
delivery conditions. They do not establish an unsupervised chatbot's efficacy or
make Luna equivalent to practitioner-supported guided self-help.

## 2. Uphoff et al. (2020): behavioural activation

**Source:** [Cochrane review CD013305](https://www.cochrane.org/evidence/CD013305_behavioural-activation-therapy-depression-adults), DOI: 10.1002/14651858.CD013305.pub2.

**Methods/results:** 53 studies, 5,495 participants. Short-term efficacy versus
usual care favored BA: RR 1.40, 95% CI 1.10–1.78; seven RCTs, n=1,533,
moderate certainty. The advantage did not persist in worst-case or intention-to-
treat sensitivity analyses. Against CBT: RR 0.99, 95% CI 0.92–1.07; five RCTs,
n=601, moderate certainty. Absence of a detected difference is not proof of
equivalence. Most findings were short-term.

**Verified excerpt, plain-language summary:** “activities which are meaningful to them”

**Interpretation for Luna:** Consider meaningful, feasible engagement and monitor
what happens after an activity. Distinguish interest, completion, and helpfulness.
Theoretical concern: immediate relief may sometimes reinforce avoidance, so mood
improvement alone is an incomplete optimization target.

**Boundary:** This is a multicomponent intervention literature, not validation of
individual videos or an emotion-to-activity lookup. Full review methods and
trial-level risk-of-bias details remain unread. Evidence certainty is mostly low
to moderate across comparisons.

## 3. Braun et al. (2015): purposeful Socratic questioning

**Source:** [Full manuscript, PMC4449800](https://pmc.ncbi.nlm.nih.gov/articles/PMC4449800/), DOI: 10.1016/j.brat.2015.05.004.

**Methods/results:** Of 67 outpatients entering a 16-week cognitive-therapy study,
55 met the early-session data requirements. Four raters assessed each recorded
session using a five-item questioning scale. Analyses separated detrended
within-patient variation from between-patient differences and controlled current
symptoms. One SD higher within-patient questioning predicted a 1.51-point lower
next-session BDI-II score (p=.01); 1.49 points after alliance adjustment.

**Verified excerpt, Discussion:** “we cannot definitively establish a causal relation of Socratic questioning and outcome without an experimental manipulation.”

**Interpretation for Luna:** Evaluate questions that explore central appraisals
and alternative perspectives. Generic question frequency is insufficient. A
question should create a useful distinction or address a genuine information gap.

**Boundary:** Time-varying confounding remains possible. Analysis concerned early
sessions, excluded insufficient-data cases, and used a predominantly White
clinical sample. The alliance Relationship subscale had ICC=.50. Cognitive change
was proposed as a mechanism but not tested here; controlling alliance does not
establish causal independence from every relational process.

## 4. Lieberman et al. (2007): affect labeling

**Source:** [Putting feelings into words](https://doi.org/10.1111/j.1467-9280.2007.01916.x).

**Methods/results, abstract only:** fMRI during affect labeling showed diminished
amygdala/limbic responses relative to other encoding conditions, increased right
ventrolateral prefrontal activity, and an inverse RVLPFC–amygdala association
statistically mediated by medial prefrontal activity.

**Verified excerpt, abstract:** “the mechanisms by which affect labeling produces this benefit remain largely unknown”

**Interpretation for Luna:** Optional labels may support semantic categorization
of affect. This is a hypothesis to test, particularly because AI-suggested labels
can anchor a user's interpretation. Preserve self-generated labels, uncertainty,
and correction.

**Boundary:** Lower amygdala BOLD does not establish lower subjective distress or
clinical benefit. Statistical mediation is not experimental proof of a directed
RVLPFC → MPFC → amygdala inhibition mechanism. Language, attention, and task
demands require examination in the full methods. Assigning labels to one's own
experience differs from an AI inferring feelings from writing. The sample and
exact contrasts have not been extracted from full text.

## 5. Webb, Miles, and Sheeran (2012): strategy-specific regulation

**Source:** [Dealing with feeling](https://doi.org/10.1037/a0027600).

**Verified excerpt, abstract:** “306 experimental comparisons of different emotion regulation (ER) strategies”

**Results, abstract only:** Pooled effects differed across processes: attentional
deployment d+=0.00, response modulation .16, cognitive change .36. Crucially,
subtypes differed: distraction .27 versus concentration −.26; perspective taking
.45 versus reappraising the emotional response .23. Outcomes included experience,
behavior, and physiology. Strategy effectiveness had multiple moderators.

**Interpretation for Luna:** Distinguish the intended process of an activity.
Redirecting attention, considering an alternative appraisal, and changing overt
expression are different interventions. Evaluate whether each fits the user's
goal rather than calling all resources generic coping.

**Boundary:** These pooled estimates are not probabilities of helping an
individual, head-to-head rankings for Luna, or necessarily comparable real-world
effects. An aggregate null can conceal opposing subtype effects. Full coding
rules, confidence intervals, heterogeneity, and moderator analyses remain unread.
Do not turn reappraisal into pressure to deny valid concerns.

## 6. Bonanno and Burton (2013): regulatory flexibility

**Source:** [Regulatory flexibility](https://doi.org/10.1177/1745691613504116).

**Evidence, abstract only:** A conceptual review challenging the assumption that
regulation strategies are uniformly beneficial or maladaptive. It proposes a
heuristic framework with three components.

**Verified excerpt, abstract:** “sensitivity to context, availability of a diverse repertoire of regulatory strategies, and responsiveness to feedback”

**Interpretation for Luna:** This is the most directly useful organizing
hypothesis for recommendations: assess context, present plausible alternatives,
and revise after feedback. Confirmed emotion alone does not describe what the
person wants, what is controllable, or which options are feasible.

**Boundary:** A conceptual framework does not establish intervention efficacy or
validate a recommender. Greater strategy switching is not automatically better
regulation. Operationalize context sensitivity and feedback responsiveness before
claiming Luna implements psychological flexibility. The full review's evidence
and methodological discussion remain unverified.

## 7. Ryan and Deci (2000): autonomy-supportive interaction

**Source:** [Self-determination theory and the facilitation of intrinsic motivation, social development, and well-being](https://doi.org/10.1037//0003-066X.55.1.68).

**Evidence, abstract only:** A foundational synthesis examining social-contextual
conditions that support or undermine intrinsic motivation, self-regulation, and
wellbeing. It proposes three psychological needs.

**Verified excerpt, abstract:** “competence, autonomy, and relatedness”

**Interpretation for Luna:** In SDT, autonomy concerns volitional endorsement,
not merely the number of available choices. Explain recommendations, make them
editable and rejectable, and avoid pressuring users to continue. Feasible actions
may support competence. These are theory-informed product hypotheses; the
operational details are not established by this abstract.

**Boundary:** Ranking options measures preference, and post-use ratings measure
reported experience. Neither establishes need satisfaction. Relatedness should
not be conflated with simulated attachment to the assistant. The full paper and
later digital-intervention evidence remain to be reviewed.

## A relevant follow-up found during the search

[Vittorio et al. (2022)](https://doi.org/10.1016/j.brat.2022.104035), *Using Socratic
Questioning to promote cognitive change and achieve depressive symptom reduction*,
was found through Europe PMC. Its abstract reports n=123 CBT clients, a significant
indirect association through cognitive change, and stronger questioning–cognitive
change association in clients with lower pretreatment CBT skills. This directly
addresses the mechanism left untested in Braun et al. It is abstract-only evidence
here; mediation alone does not settle causality. No effect estimate was provided
in the retrieved abstract, and none is invented in this report.

## Proposed product model and evaluation

Treat the following as our design hypothesis, not a validated protocol:

**Clarify the user's goal → identify one useful question or distinction → offer
correctable emotion labels when useful → propose context-sensitive options →
respect the choice → collect feedback after use.**

Do not prescribe questions for every turn. Direct-answer requests, listening
preferences, corrections, and stopping remain controlling inputs. The papers do
not establish that one question per turn is an optimum; that is a usability
constraint to evaluate. An emotion-to-activity table would miss the contextual
information these hypotheses require.

Evaluate focused questioning against the current reflective style for perceived
insight, grounding, burden, and invalidation. Evaluate emotion suggestions for
usefulness, anchoring, and correction. Evaluate recommendations separately for
fit, feasibility, selection, completion, and post-use helpfulness. Preference and
immediate mood are incomplete reward signals. Pre/post mood changes without a
comparison condition cannot establish causal benefit.

NICE and Cochrane provide clinical evidence anchors; the process and laboratory
studies suggest mechanisms to investigate; the conceptual reviews organize
design. None directly validates the proposed combined LLM experience. Broader,
more recent systematic reviews and direct conversational-agent studies remain a
separate evidence gap.

## Retrieval and quotation audit

Public source snapshots, metadata, and an exact-match quotation check are saved
under `/workspace/journalpulse-planning/psychology-evidence/`. Key records:

- `nice-browser.txt`: relevant recommendations inspected.
- `cochrane.txt`, `activation-meta.txt`: official summary and indexed abstract.
- `socratic-ncbi-xml.raw`: full manuscript via NCBI efetch for PMC4449800.
- `epmc-search.txt`, `regulation-meta-api.txt`, `flexibility-meta.txt`,
  `sdt-meta.txt`: indexed abstracts.
- `socratic-search.txt`: original and 2022 follow-up metadata/abstracts.
- `verified-quotes.json`: seven exact quotations, each under 25 words.

NCBI's XML response for PMC7390059 contained front matter/abstract only; it is not
counted as a full review. HTTP 200 browser-challenge pages were also excluded.
Publisher challenges and unavailable routes were not bypassed. No journal data,
paid model calls, search-provider calls, or deployment changes were involved.
