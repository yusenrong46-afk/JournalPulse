# Luna: evidence for a chat–action–feedback experience

Reviewed October 5, 2026. This is a research and product-design proposal.
No application behavior, prompt, skill, database, or deployment was changed.

## Question and reading scope

Can Luna understand a user's intent, proactively suggest an activity, negotiate
the suggestion, and use reported outcomes to adapt what happens next?

This focused review examined full-text methods, results, and limitations for
Balban et al. (2023), both Fincham et al. (2023) papers, von Lützow et al. (2025),
and Jaremba et al. (2026), plus the full conceptual paper by Nahum-Shani et al.
(2018). Weinstein et al. (2024) was accessible as an indexed abstract only.
Previously reviewed Cochrane behavioral-activation evidence is reused at its
original summary/abstract level. Supplements and raw trial datasets were not
independently audited. This is not a systematic literature review.

## Research findings and design consequences

### 1. Brief breathing practices: preliminary randomized evidence

[Balban et al., 2023, Cell Reports Medicine](https://doi.org/10.1016/j.xcrm.2022.100895)
— *Brief structured respiration practices enhance mood and reduce physiological arousal*.

The remote study enrolled 108 adults in mindfulness meditation or one of three
breathing conditions, practiced for five minutes daily over 28 days. Allocation
was initially randomized in blocks, with household members assigned together.
Daily pre/post positive affect, negative affect, and state anxiety were measured;
wearables provided daily physiological measures.

Breathwork, particularly cyclic sighing, showed greater improvement in positive
affect than mindfulness meditation. Respiratory rate declined more in the
breathwork group. Anxiety and negative-affect changes did not significantly differ
between the combined breathing and meditation conditions. Neither resting heart
rate nor HRV showed significant change across conditions. Trait anxiety and sleep
outcomes did not establish an advantage.

**Interpretation:** Brief activities can be investigated for immediate subjective
effects. Physiological and psychological outcomes must be kept separate; this
study does not establish vagal mediation or a general nervous-system reset.

**Limits:** Exploratory and retrospectively registered, small groups, mostly
university recruitment, remote fidelity uncertain, and no longer-term follow-up.
The specific five-minute protocols cannot validate arbitrary video recommendations
or a different activity duration.

**Design consequence:** A brief activity is a plausible candidate, presented with
an expected goal and an invitation to report the actual effect.

### 2. Breathwork meta-analysis: modest average benefit

[Fincham et al., 2023, Scientific Reports](https://doi.org/10.1038/s41598-022-27247-y)
— *Effect of breathwork on stress and mental health*.

The primary stress analysis included 12 randomized trials and 785 adults. The
pooled effect was Hedges g=-.35, 95% CI -.55 to -.14, with I²=42%, compared with
non-breathwork controls. Most studies had moderate risk of bias; follow-up evidence
was limited. Secondary anxiety and depression analyses were not appraised as
thoroughly as the primary outcome.

**Interpretation:** There is support for considering breathwork among options.
The pooled result does not identify a best technique for an individual, establish
the best moment to offer it, or prove an immediate effect from one session.

**Design consequence:** Include suitable, reviewed activities with alternatives;
avoid treating one body-based technique as the default answer to every emotion.

### 3. A stronger active-control test tempers technique-specific claims

[Fincham, Strauss, and Cavanagh, 2023, Scientific Reports](https://doi.org/10.1038/s41598-023-49279-8)
— *Effect of coherent breathing on mental health and wellbeing*.

A preregistered, participant-blinded trial randomized 400 adults to approximately
5.5 breaths/minute or a credible matched 12-breaths/minute comparison, for about
10 minutes/day over four weeks. Credibility and expected benefits did not differ.
Stress improved over time in both groups, but the primary group-by-time test did
not show superiority of slow breathing (p=.765). Secondary outcomes similarly
did not establish superiority.

**Interpretation:** A plausible physiological explanation and improvement after
an activity do not demonstrate a specific causal mechanism. Attention, taking a
break, expectancy, shared breathing features, and natural change remain possible
contributors. The active comparison may itself help. This trial measured outcomes
over weeks and does not rule out brief state changes immediately after a session.

**Design consequence:** User feedback is useful for choosing the next step but
must not be presented as proof that Luna or a particular technique caused change.

### 4. Acute exercise: short-term affect changes, variable response

[Weinstein et al., 2024, Psychosomatic Medicine](https://doi.org/10.1097/PSY.0000000000001321)
— *Affective Responses to Acute Exercise*.

The abstract reports 103 studies and 4,671 participants. Mood assessed before and
within 30 minutes after one exercise bout improved on average (g=.336, 95% CI
.234–.439); anxiety and depressive-symptom measures also improved. Heterogeneity
was substantial and remained unexplained by the examined moderators.

**Limits:** Abstract-only. These are reported pre/post effects; this review does
not treat them as a pooled randomized comparison against no exercise. The exact
protocol distribution, bias appraisal, and best dose remain unread.

**Design consequence:** Movement is a reasonable option to test when compatible
with the person's circumstances and preferences. No fixed five-minute walk or
emotion-to-exercise mapping is established by this abstract.

### 5. JITAI: the closest scientific design framework

[Nahum-Shani et al., 2018, Annals of Behavioral Medicine](https://doi.org/10.1007/s12160-016-9830-8)
— *Just-in-Time Adaptive Interventions in Mobile Health*.

This conceptual paper describes an intervention that adapts the type, timing,
and amount of support to changing context. It distinguishes six elements:
long-term outcomes, short-term outcomes, decision points, intervention options,
tailoring variables, and decision rules.

Receptivity means the person's current willingness and capacity to use support.
Need and receptivity are different. The framework explicitly includes offering
no intervention when help is unnecessary or poorly timed. Repeated assessment
itself can be burdensome. It also discusses micro-randomized trials as a way to
test the effects of particular options at repeated decision points.

**Interpretation:** This is a useful architecture for our proposed loop, not a
trial demonstrating its efficacy. Adding an LLM does not automatically validate
its timing or recommendations. Luna can be JITAI-inspired without passive sensing.

**Design consequence:** Each eligible chat turn can be a decision point. Use the
user's stated goal, constraints, last reported state, preferences, and feedback
to choose between clarifying, reflecting, offering an action, checking an outcome,
or pausing. The user need not choose the activity category before Luna recommends.

### 6. Adaptive mobile interventions: small controlled effects

[von Lützow, Neuendorf, and Scherr, 2025, BMJ Mental Health](https://doi.org/10.1136/bmjment-2025-301641)
— *Effectiveness of just-in-time adaptive interventions for improving mental
health and psychological well-being*.

The review included 23 studies and 2,563 recruited participants. The main
between-group analysis used 22 studies and found g=.15, 95% CI .05–.26, I²=20%.
The authors rated overall mental-health evidence low certainty. The review mixes
JITAIs and less-adaptive ecological momentary interventions; definitions and
decision rules were inconsistently reported. Adherence and missing outcomes were
major bias concerns.

**Important appraisal:** The larger follow-up estimates are baseline-to-follow-up
changes within intervention groups, rather than the same controlled estimand as
the main g=.15. They should not be described as growing causal treatment effects.
Significant effects in one control subgroup but not another do not establish a
difference between subgroups; the reported subgroup-difference test was not
significant. No included comparison establishes the value of Luna's LLM policy.

**Design consequence:** Adaptive support is worth testing, but benefits should be
measured rather than inferred from personalization language or more engagement.

### 7. A recent study shows why timing and burden matter

[Jaremba et al., 2026, Scientific Reports](https://doi.org/10.1038/s41598-026-39518-z)
— *Optimizing just-in-time adaptive interventions for interpersonal distress*.

Secondary analysis of a feasibility randomized trial included 77 participants,
largely university students, receiving mindfulness or mentalization suggestions.
About 79.3% of triggered suggestions were not engaged with. Higher stress was
associated with greater non-engagement. No significant proximal changes were
detected at the next assessment, approximately two hours later; all reported
proximal effect magnitudes were below |d|=.06.

**Limits:** Engagement and timing were not themselves randomized. Treatment-
confounder feedback, selection effects, and the assessment window restrict causal
inference. The null result cannot rule out a shorter-lived immediate effect.
Network associations do not establish directed neurocognitive mechanisms.

**Design consequence:** Detecting distress is insufficient justification for
repeated recommendations. Use few steps, a feasible activity, a clear decline
option, and appropriate feedback timing. Do not interpret an ignored suggestion
as proof the user needs a more insistent coach.

### 8. Behavioral activation keeps the goal broader than immediate mood

[Uphoff et al., 2020, Cochrane](https://doi.org/10.1002/14651858.CD013305.pub2)
— *Behavioural activation therapy for depression in adults*.

Previously reviewed official summary/abstract: 53 studies, 5,495 participants.
The intervention literature supports investigating meaningful activity, with
certainty and robustness varying by comparison. It concerns structured treatment
over time, not isolated links or guaranteed instant relief.

**Design consequence:** Track useful engagement with the user's goal as well as
short-term state. A meaningful step can be worthwhile while some anxiety remains.
Conversely, immediate relief alone does not establish that an activity helps with
the concern the user brought.

## Current experience, checked in the source

The current talk page has an action invitation, then feelings selection, goal
selection, and a catalog list. Search opens the separate discovery flow. There is
already a separate completion/helpfulness check-in with optional feelings and a
note. Basic activity and outcome infrastructure exists.

Source anchors: `web/app/talk/page.tsx` around lines 947–1038;
`web/app/check-in/page.tsx` around lines 130–186; `src/journalpulse/resources.py`
`approved_actions` filters the catalog by intent/style/goal and returns its first
eight matches. The proposals below are behavior changes, not existing capabilities.

## Proposed experience

| Moment | Current flow | Proposed flow |
| --- | --- | --- |
| Explain the concern | Chat | Warm chat; use the user's stated intent |
| Identify a target | Separate feelings and goal steps | Infer a tentative target from conversation; ask only for an essential missing detail |
| Get a suggestion | User enters the activity flow and chooses from a list | Luna proposes one relevant activity with reason, duration, and effort |
| Negotiate | Change selection or enter discovery | Reply naturally: shorter, seated, no audio, different approach, or keep talking |
| Find a resource | Separate search flow | Inline discovery when needed, using the existing consented Brave capability |
| Try | Open activity/timer | A clear activity state with pause/stop and return to the same chat |
| Report | Separate check-in | Brief inline report: tried, not yet, stopped; then effect on the user's target |
| Adapt | Outcome saved for review | Luna uses the reported result to continue, revise, reflect, or finish |

### Recommendation policy

The practical inputs are the user's desired outcome, relevant constraints,
confirmed preferences, latest reported state, and recent accepts/rejections.
An inferred emotion is a hypothesis, not a confirmed self-report or sensor reading.

Luna may proactively offer an action when it has enough context for a reasonable
proposal and no controlling request to avoid advice or activities. It should
ask one essential clarification when fit cannot be judged. Listening and stopping
remain valid decisions. No distress threshold or fixed number of chat turns
should force an action.

One inline card shows the action, a short reason, duration/effort, and source where
applicable. Primary choices: Try it, Adjust it, Keep talking. Free-text negotiation
is always possible. These are proposed controls, not research-proven labels.

### Brave's role

Use a reviewed catalog option when it fits. Search when a suitable option is
missing or the user wants another resource. Use an editable general activity
query under the existing search consent, without journal passages or private
conversation details. Carry forward constraints and rejected links.

Search finds resources; an activity study does not validate an arbitrary web
page. Apply source/content criteria and show provenance. The current backend
uses snippets, so do not imply that full linked content has been clinically
reviewed. Clicking a link is neither completion nor permission to save it.

### Feedback and adaptation

Record what the user actually tried before asking about effect. A brief report
can be conversational or a few choices: closer to the desired state, unchanged,
further away, or unsure. Make an optional same-scale before/after rating available
for the chosen target, not mandatory emotion questionnaires.

Separate three questions conceptually: Did it fit? Did they try it? What changed?
Declining, being unable to try, and trying without benefit are different outcomes.
Check near the relevant time; a short relief activity may warrant an immediate
check, while a social or longer activity may be discussed later. Reminders require
the user's choice and appropriate implementation.

After feedback, Luna can acknowledge a useful result, revise an unsuitable
suggestion, return to the concern, or end the session. It should not automatically
start another recommendation cycle. Personalization from repeated reports should
remain tentative; one positive experience is not an established individual effect.

## First vertical slice and evaluation

Build one complete in-chat loop using existing activity and outcome infrastructure:
understand, propose, negotiate, try, report, adapt. Reuse Brave for a bounded
alternative-resource path rather than introduce new search infrastructure.

First freeze fictional conversations and compare current behavior with the new
instructions for fit, grounding, recommendation timing, correction handling,
search provenance, refusal, and stopping. Include a useful-action case, an
unsuitable first suggestion, no improvement, worse discomfort, a positive
experience needing no intervention, and a user who wants only to talk.

Then test with people for comfort, effort, perceived relevance, helpfulness, and
progress toward their chosen goal. Record immediate state separately from later
goal progress. Small usability sessions and before/after ratings are exploratory;
they do not establish clinical efficacy. A later controlled comparison, and
eventually a properly designed micro-randomized study, could test whether
recommendation timing and adaptation add value.

The initial endpoint is a useful, manageable step or clearer understanding at the
user's chosen stopping point. Longer conversations, more clicks, and accepting
more recommendations are incomplete success measures.
