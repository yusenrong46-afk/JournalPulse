# Evidence review for Luna's first reflection skill

Reviewed October 5, 2026. Research only: no application instructions, skill,
evaluation conversations, provider calls, or deployment changed.

This is a focused evidence review, not a systematic review. The question is:
which conversational principles are defensible for a small, user-controlled
reflection feature, and what must be tested locally?

## Reading scope and evidence hierarchy

Full-text methods, results, and discussion were examined for Braun et al. (2015),
Magill et al. (2018), Roh et al. (2026), Zhang et al. (2025), and Huang et al.
(2026). Supplementary datasets, analysis code, and all appendices were not audited.
Vittorio et al. (2022), Elliott et al. (2018), and Ng et al. (2012) were reviewed
through indexed abstracts only. Their complete methodological quality therefore
remains unassessed here.

Clinical process studies suggest mechanisms but do not randomize individual
conversational techniques. Meta-analyses increase breadth but cannot make
correlational process estimates causal. Direct chatbot evidence is closer to our
delivery medium, but different systems, populations, and objectives limit transfer.
None evaluates JournalPulse, GPT-6 Luna, our proposed prompt, or a Markdown skill.

## 1. Braun et al. (2015): guided discovery in cognitive therapy

[Therapist use of Socratic questioning predicts session-to-session symptom change
in cognitive therapy for depression](https://doi.org/10.1016/j.brat.2015.05.004).
[Full manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC4449800/).

**Design and result.** Of 67 depressed outpatients entering a 16-week cognitive
therapy study, 55 had sufficient early-session observations for the analysis.
Observer-rated questioning in sessions 1–3 was separated into detrended within-
and between-patient components. A one-standard-deviation increase in within-
patient questioning predicted a 1.51-point lower next-session BDI-II score
(p=.01), or 1.49 points after adjustment for two alliance components.

**Why it matters.** Focusing on within-person variation addresses stable patient
confounding better than a simple across-patient correlation. The rating scale
included exploration of alternative perspectives and emotionally central
cognitions. The candidate mechanism is reconsidering the interpretation attached
to an event, rather than asking generic questions merely to continue a dialogue.
The scale also includes question quantity; this paper does not independently
identify which question properties drive the association.

**Limit.** Questioning was not randomized; time-varying confounding and response-
dependent therapist behavior remain plausible. The sample was predominantly White,
and only early sessions were examined. Cognitive change itself was not tested as
a mediator. No evidence here establishes an optimum of one question per turn.

**Luna hypothesis.** Explore a relevant distinction between event and interpretation
when the user wants reflection; do not assume the interpretation is wrong.

## 2. Vittorio et al. (2022): a candidate cognitive mechanism

[Using Socratic Questioning to promote cognitive change and achieve depressive
symptom reduction](https://doi.org/10.1016/j.brat.2022.104035).

**Abstract-level finding.** In 123 CBT clients, cognitive change statistically
mediated the association between questioning and early symptom change. The
questioning–cognitive-change relationship was stronger for clients with lower
pretreatment CBT skills. No numerical indirect-effect estimate is supplied in the
retrieved abstract.

**Why it matters.** It makes cognitive change a plausible process measure beyond
conversation length or pleasantness. It also suggests that users may need
different amounts of support.

**Limit.** Statistical mediation does not by itself establish a causal chain.
Full timing, measures, effect estimates, and alternative models remain unread.
Do not infer that Luna should measure or diagnose a user's CBT skill level.

**Luna hypothesis.** Look for the user articulating a clearer or revised
understanding in their own words, without rewarding agreement with Luna.

## 3. Magill et al. (2018): reflective listening and ambivalence

[A meta-analysis of motivational interviewing process: Technical, relational, and
conditional process models of change](https://doi.org/10.1037/ccp0000250).
[Full manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC5958907/).

**Design and result.** 58 reports described 36 primary studies, 40 effect sizes,
and 3,025 participants, mostly in alcohol/drug behavior-change contexts. The
bundle of MI-consistent skills, including open questions, reflections, and
affirmations, correlated with both change talk (r=.55) and sustain talk (r=.40).
Proportion change talk correlated with lower subsequent risk behavior (r=-.16).
Change-talk frequency alone did not predict outcome. The reflection-to-question
ratio did not significantly predict proportion change talk (r=.03, p=.281).

**Why it matters.** Productive discussion may expose both sides of ambivalence.
Getting users to agree, accept advice, or say more positive things is an
incomplete success criterion. Counting reflections or questions cannot substitute
for evaluating their function. The bundled estimate does not isolate the effect
of reflective listening itself.

**Verified quotation:** “MI should facilitate an atmosphere where both positive
and negative aspects of behavior change can be safely examined.”

**Limit.** Observational process correlations, mostly single/early sessions and
behavior-change populations. The authors explicitly reject individual causal
inferences from the meta-analysis. A small complex-reflection association was
sensitive to excluding a study with weak rating reliability. Global empathy did
not significantly predict outcomes within this particular process literature;
restricted therapist variability limits interpretation.

**Luna hypothesis.** Reflect concisely, allow mixed feelings, and avoid imposing
an action goal when the user wants understanding. Do not adopt a fixed
reflection-to-question ratio as scientifically established.

## 4. Elliott et al. (2018): empathy and perceived understanding

[Therapist empathy and client outcome: An updated
meta-analysis](https://doi.org/10.1037/pst0000175).

**Abstract-level finding.** 82 independent samples and 6,138 clients yielded a
weighted empathy–outcome correlation r=.28, 95% CI .23–.33, with considerable
heterogeneity. Perception measures predicted outcomes better than empathic
accuracy measures.

**Why it matters.** Keep responsiveness and understanding while adding structure.
This broader psychotherapy literature should not be treated as contradicted by
the narrower null empathy result in the MI process review: their populations,
measurement ranges, and endpoints differ.

**Limit.** Association is not proof that increasing scripted empathy improves
outcomes. Perceived understanding can coexist with inaccurate interpretations.
Full risk-of-bias and moderator details remain unread.

**Luna hypothesis.** Measure both whether users feel understood and whether Luna
actually stays grounded. Pleasantness alone is insufficient.

## 5. Ng et al. (2012): autonomy support

[Self-Determination Theory Applied to Health Contexts: A
Meta-Analysis](https://doi.org/10.1177/1745691612447309).

**Abstract-level finding.** Synthesis of 184 independent datasets reported
associations among practitioner autonomy support, psychological need satisfaction,
autonomous motivation, and beneficial health outcomes; meta-analyzed correlations
were also examined with path analysis.

**Why it matters.** Autonomy concerns volitional endorsement, not simply the
number of buttons offered. The user's goal and permission to reject the
assistant's interpretation are central design considerations.

**Limit.** Abstract-only here; no numerical pooled estimate is invented. These
relations do not establish that a stop button or optional question causes need
satisfaction or health improvement.

**Luna hypothesis.** Allow correction, skipping, changing direction, direct-answer
requests, and stopping. Do not optimize for persuading users to continue.

## 6. Roh, Yoon, and Oh (2026): direct evidence on exploratory chatbot dialogue

[SleepPathfinder: A Socratic Questioning and Self-Decision–Based Chatbot to Support
User Engagement in Digital CBT-I](https://doi.org/10.2196/79242).
[Full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13249113/).

**Design and result.** A formative pilot included 45 participants. A subsequent
five-day comparison assigned 30 university participants to exploratory or
directive dialogue, 15 per condition. The manuscript does not clearly describe
random allocation; this review therefore does not label it an RCT. Exploratory
dialogue reduced perceived severity of sleep problems relative to the directive
condition (reported interaction p=.02; Hedges g=.808). No significant interaction
was found for behavioral intention. Autonomy and hedonic-quality differences did
not meet conventional significance thresholds (p=.07 and .18).

**Important tradeoff.** More causal self-explanation was associated with greater
appraisal change; more question-tagged chatbot turns were associated with smaller
change. Early session termination was more frequent in the exploratory condition
(reported rate .446 versus .000). The paper's group-level question metric was
actually higher in the directive condition, so these associations cannot simply
be read as effects of the assigned exploratory condition.

**Why it matters.** This directly tests a dialogue design close to our hypothesis,
but does not show that exploratory dialogue is unambiguously better. It identifies
burden and premature stopping as outcomes worth inspecting alongside usefulness.

**Verified quotation:** “rather than aggressive questioning strategies in
conversational support systems”

**Limit.** Small, short, exploratory, multiple outcomes; no clinical sleep outcome
was assessed. Self-awareness increased without a significant between-group
difference. Causal-word counts are proxies, not validated measures of insight.
The system and sleep-specific objective differ from Luna's general reflection.

**Luna hypothesis.** Use questions selectively, offer concise summaries, and
evaluate whether users feel pressured or stuck in an interview. A user-requested
stop remains a successful interaction, not a failure to retain engagement.

## 7. Zhang et al. (2025): broader generative-chatbot evidence

[Generative AI Mental Health Chatbots as Therapeutic Tools: Systematic Review and
Meta-Analysis](https://doi.org/10.2196/78238).
[Full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12707440/).

**Design and result.** 26 studies in the narrative synthesis and 14 RCTs reported
in the main meta-analysis, N=6,314. Reported average effect size .30, 95% CI
.004–.59; prediction interval -.85–1.67. The average is modest and highly
uncertain for a new setting. None of the 14 RCTs was rated low risk of bias:
nine had some concerns and five high risk. About 69% of the 26 interventions
included some human assistance.

**Why it matters.** It supports investigating conversational interventions,
while showing why we cannot transfer efficacy to a prompt-only autonomous app.
The review excluded positive wellbeing outcomes from the meta-analysis, so it
does not establish benefits for positive-event reflection.

**Limit and reporting concern.** Registration was retrospective. The limitations
section refers to 12 studies, conflicting with the main reported 14; a table note
also mixes article and treatment-pair counts. Those inconsistencies and baseline/
attrition concerns reduce confidence in precise quantitative interpretation.
Study-level social-vs-task moderator findings are not randomized comparisons of
prompt components.

**Luna hypothesis.** Measure conversational usefulness and grounding first;
do not advertise symptom reduction from the before/after prompt experiment.

## 8. Huang et al. (2026): scrutinizing interaction-feature claims

[Therapeutic Interaction Features of AI Chatbots in Depression Interventions:
Systematic Review and Meta-Analysis](https://doi.org/10.2196/88697).
[Full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13318397/).

**Design and result.** 11 RCTs and 2,220 participants were used for the symptom
analysis. The HKSJ-adjusted pooled SMD was -.46, 95% CI -1.02–.10, with substantial
heterogeneity (I²=87%). The authors rated symptom evidence very low certainty.
Adherence was also not significantly improved overall. Interaction features were
rated from published descriptions, not by inspecting actual chatbot turns.

**Reporting concern.** The results paragraph gives p=.01 alongside the adjusted
CI crossing zero. The discussion clarifies that the conservative HKSJ result
was not significant. We use that cautious interpretation, not the isolated
p-value. Subgroup significance in one group and nonsignificance in another does
not by itself establish a difference between them.

**Why it matters.** Useful as a warning against treating engagement, warmth, and
depth as proven causal treatment ingredients. The feature findings generate
hypotheses; they do not justify a prescribed dialogue recipe.

## What is defensible for the first slice

Our design hypothesis is **grounded, autonomy-supportive guided reflection**:
understand the requested goal; decide whether a question, direct answer, reflection,
or summary is useful; explore one unresolved distinction; make interpretations
tentative; accept corrections; stop when asked.

This is our proposed synthesis, not a validated combined CBT/MI protocol.
One question per turn is a usability constraint to test, not an evidence-based
dose. Do not frame normal emotions as cognitive distortions or force positive
reappraisal. More disclosure, longer chats, advice acceptance, and user agreement
are not sufficient measures of success.

Before writing the skill, define evaluation criteria for useful clarification,
grounding, perceived understanding, correction handling, autonomy, naturalness,
and questioning burden. Test positive experiences as well as distress because
the clinical literature does not automatically cover that use case. Controlled
before/after conversations can test these behaviors; they cannot establish
clinical efficacy or the proposed cognitive mechanism.

## Retrieval record

Public search results and source snapshots are in this directory, with the
`slice-` prefix. Braun's full manuscript was previously retrieved and re-examined
here; its indexed metadata was checked again. Magill's manuscript was retrieved
from NCBI BioC XML. The three newer full articles were saved as valid XML response
bodies by the browser before its subsequent body-text locator timed out; their
XML bodies were parsed and read locally. The locator errors do not mean those
retained article bodies were inaccessible. Their supplementary files were not
retrieved.

Therabot's published publisher page was inaccessible in this run. A retrieved
preprint was not substituted for a verified published RCT, and no Therabot effect
claim is used in this review. Publisher challenges and network restrictions were
not bypassed. HTTPS verification remained enabled.
