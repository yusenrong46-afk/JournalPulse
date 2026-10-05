# Psychology evidence: historical provisional notes

**Superseded in part by [the browser-verified research summaries](PSYCHOLOGY_RESEARCH_SUMMARIES.md).**
Browser retrieval subsequently succeeded for relevant NICE recommendations,
Cochrane's summary/abstract, the full Braun et al. manuscript, and the other
papers' indexed abstracts. The access failures below describe the earlier HTTP
attempts. Use the newer report for verified quotations and precise reading scope;
the remaining full-text limitations still apply.

Prepared October 5, 2026. Status: provisional source shortlist and research plan,
not a completed online literature review. No exact quotations are approved yet.

## Access limitation and work completed

Public requests to PubMed/PMC, NICE, Cochrane, Europe PMC, and Crossref were
blocked by the environment's egress proxy (HTTP 403). A network permission grant
did not change the proxy allowlist. Research-domain additions were saved to the
cloud configuration draft while preserving the existing application domains;
the tool reported `requires_publish: true`. A subsequent NICE request remained
blocked. No paper, abstract, or guideline was successfully retrieved this turn.

References below are identified from existing knowledge. Verify the bibliographic
details, source text, current guideline version, and applicable findings online
before treating them as the upgrade's evidence basis or quoting them. The
application, Vercel deployment, provider settings, and database were not changed.

## Proposed source shortlist

| Source to verify | Evidence type and relevance | Proposed design hypothesis, not a verified finding from this review |
| --- | --- | --- |
| [NICE NG222: Depression in adults—treatment and management](https://www.nice.org.uk/guidance/ng222/chapter/Recommendations) | Clinical guideline; inspect shared decision-making and descriptions of CBT/behavioural activation. It applies to depression treatment, not a general journaling chatbot. | Respect user preferences and use structured approaches without representing Luna as equivalent to clinician-delivered treatment. |
| Uphoff et al. (2020), [Behavioural activation therapy for depression in adults](https://doi.org/10.1002/14651858.CD013305.pub2), Cochrane Database of Systematic Reviews | Systematic review; inspect comparative outcomes, certainty, populations, intervention delivery, and follow-up. | Offer feasible, meaningful activities and assess helpfulness after use; do not assume each emotion has one clinically correct activity. |
| Braun et al. (2015), [Therapist use of Socratic questioning predicts session-to-session symptom change in cognitive therapy for depression](https://pubmed.ncbi.nlm.nih.gov/?term=Therapist+use+of+Socratic+questioning+predicts+session-to-session+symptom+change) | Therapy-process study; verify design and causal limits. | Test collaborative, focused questions rather than repetitive validation or interrogative questionnaires. Therapist evidence does not establish chatbot efficacy. |
| Lieberman et al. (2007), [Putting feelings into words: affect labeling disrupts amygdala activity in response to affective stimuli](https://doi.org/10.1111/j.1467-9280.2007.01916.x), Psychological Science | Laboratory/neuroimaging study; inspect task, sample, and measured outcomes. | Let users label or correct their emotions. Neural findings do not establish that an AI-generated emotion label improves daily wellbeing. |
| Webb, Miles, and Sheeran (2012), [Dealing with feeling: a meta-analysis of the effectiveness of strategies derived from the process model of emotion regulation](https://doi.org/10.1037/a0027600), Psychological Bulletin | Meta-analysis; inspect strategy definitions, moderators, outcomes, and heterogeneity. | Evaluate different ways of responding rather than assuming one universal emotional-regulation strategy. |
| Bonanno and Burton (2013), [Regulatory flexibility: An individual differences perspective on coping and emotion regulation](https://doi.org/10.1177/1745691613504116), Perspectives on Psychological Science | Conceptual review; inspect contextual sensitivity, available strategies, and feedback. | Recommendations should consider context, constraints, and feedback as well as emotion labels. A conceptual framework does not validate an implemented ranking algorithm. |
| Ryan and Deci (2000), [Self-determination theory and the facilitation of intrinsic motivation, social development, and well-being](https://doi.org/10.1037/0003-066X.55.1.68), American Psychologist | Foundational theoretical/review article; verify autonomy-related arguments and supporting evidence. | Keep choices editable and optional; distinguish user agency from a claim that a specific ranking interface improves outcomes. |

## How to choose the strongest basis

Use evidence appropriate to each claim rather than name one universally strongest
paper. A high-quality clinical guideline or systematic review can anchor general
intervention principles. A therapy-process study is more directly relevant to
questioning style but weaker for causal claims. A laboratory study or conceptual
review can motivate a product hypothesis without proving everyday benefit.

Seek recent systematic reviews that update the older work, including direct
studies of conversational agents and digitally delivered interventions. Inspect
sample/population, comparison condition, effect uncertainty, attrition, adverse
events, study quality, funding, and applicability. Do not infer that a bundle of
individually motivated features constitutes a validated intervention.

## Quotation and evidence extraction protocol

For each successfully read source, record title, authors, year, DOI/permanent URL,
access date, full-text versus abstract-only access, and page/section/table. Extract
a short exact quotation, its context, and the finding's limitations. Clearly
separate quotation, plain-language paraphrase, and our product-design inference.
Do not fabricate quotations or quote search snippets as if the full paper was read.

Map every proposed behavior to a testable product hypothesis: grounded focused
questions; optional, correctable emotion suggestions; contextual activity choices;
user ranking; post-activity helpfulness. Validate those behaviors separately on
the existing evaluation set and later in a user pilot. No clinical efficacy claim
is justified by prompt compliance or by general psychology citations alone.

## Completion gate

Finish source retrieval and evidence extraction after the saved research-domain
configuration is active. Only then approve exact quotations and finalize the
psychology-informed upgrade specification. Keep the current preview unchanged.

## Academic synthesis: provisional, not a full-text review

On the user's retry request, NICE, Europe PMC, PubMed, and Crossref again returned
proxy errors. The saved draft is revision 20 and includes the research domains,
but that does not establish live access. No source text was successfully read.
The synthesis below uses established background knowledge; exact methods,
statistics, comparator details, and quotations require source verification. The
user has a neuroscience/cognitive-science background and requested this level of
analysis. Do not treat the absence of effect sizes below as a completed extraction.

### NICE NG222: guideline-level evidence

NG222 synthesizes evidence and clinical decision-making for adults with
depression; it is not a single experimental study or a chatbot protocol. Its
relevance is in organizing treatment options around clinical need, preferences,
and shared decisions. CBT and behavioural activation are structured interventions
with delivery conditions, not interchangeable collections of conversational tips.

For Luna, the defensible design hypothesis is collaborative goal formation and
transparent choice. The transfer from a guideline's clinical population and
delivery setting to a general-purpose journaling assistant is indirect. Do not
claim that citing a recommended intervention validates an LLM's implementation.
Verify current recommendations and delivery requirements before quoting them.

### Uphoff et al. (2020): behavioural activation

This Cochrane review synthesizes trials of behavioural activation for adult
depression across comparison conditions. It is an appropriate source for
clinical-outcome evidence, but comparative findings must be read with their
certainty ratings and follow-up periods. No effect-size or noninferiority claim
is made here without retrieving those details.

The underlying behavioural account concerns avoidance and reduced contact with
reinforcing experiences. Activity monitoring and scheduling aim to support
engagement with meaningful behaviour. This is more specific than distraction or
immediate mood repair. An activity can be useful without immediately improving
mood; conversely, immediate relief can reinforce avoidance.

For Luna, evaluate small activities against the user's desired outcome, time,
energy, accessibility, and values. Keep pre-activity preference separate from
completion and post-activity helpfulness. Offering a grounding video is not, by
itself, delivery of behavioural activation. Do not autonomously turn this design
into exposure treatment or infer therapeutic benefit from a click.

### Braun et al. (2015): Socratic questioning

This is a psychotherapy process-outcome study relating therapist use of Socratic
questioning to session-to-session symptom change in cognitive therapy for
depression. It supports investigating guided discovery, but it is not a randomized
comparison of Luna prompt variants. Causal interpretation depends on potential
confounding by patient engagement, alliance, therapist skill, and other process
variables; details of the study's controls need full-text verification.

A plausible cognitive account is that collaborative questions help people make
appraisals explicit, consider alternatives, and generate their own conclusions.
The intended process is not simply increasing question count. A leading question
can impose the assistant's interpretation rather than support discovery.

For Luna, ask at most one relevant question when there is a useful information gap,
and preserve direct-answer and stop behavior. Test whether users gain a grounded
distinction between event and interpretation, not merely whether the response
contains a question. Any example wording in the product specification is our
design, not a quotation from the paper.

### Lieberman et al. (2007): affect labeling

This laboratory fMRI study examined affect labeling during processing of
affective stimuli. Its commonly reported pattern includes reduced amygdala BOLD
response and increased right ventrolateral prefrontal engagement during labeling,
relative to task comparisons. Exact contrasts and sample details remain pending.

Do not equate lower amygdala BOLD with reduced anxiety or clinical improvement.
Do not infer that a particular cortical region causally inhibited the amygdala
from regional activation or statistical mediation alone. Language, attention,
categorization, and task demands are relevant alternative explanations to inspect.

For Luna, semantic labeling is a plausible component to test. However, the user
selecting an emotion label is not the same manipulation as an AI assigning an
emotion to a journal narrative. Suggested labels can also anchor users. Keep a
free-text or none-of-these option and track corrections. Emotional granularity
and long-term wellbeing are additional constructs, not established outcomes of
this one labeling experiment.

### Webb, Miles, and Sheeran (2012): emotion-regulation meta-analysis

This meta-analysis concerns strategies derived from the process model of emotion
regulation. That framework distinguishes points at which emotion can be
influenced, including attentional deployment, cognitive change, and response
modulation. The source is relevant to strategy-specific effects and moderators;
its exact pooled estimates and inclusion boundaries require verification.

Reappraisal changes the meaning assigned to a situation, while expressive
suppression targets overt expression. Neither strategy should be defined solely
by whether the user reports feeling better. Subjective experience, expressive
behaviour, and physiology are distinct outcome channels and can diverge.

For Luna, annotate activities by intended process and evaluate their fit to the
current goal. Do not convert a pooled laboratory effect into a universal ranking
for every user. Nor should reappraisal become pressure to reinterpret a genuinely
unfair situation positively. The extension to long-term real-world usefulness is
an empirical question.

### Bonanno and Burton (2013): regulatory flexibility

This is a conceptual synthesis, not a clinical trial establishing an algorithm's
efficacy. It proposes context sensitivity, access to a repertoire of strategies,
and responsiveness to feedback as important aspects of flexible regulation.

Its strongest relevance is architectural: the same named emotion can occur in
contexts with different affordances, controllability, constraints, and goals.
Consequently, emotion classification alone is insufficient for action selection.
If an attempted strategy is unsuitable, updating or stopping can be preferable to
repeating it more forcefully.

For Luna, use confirmed state plus context, goal, and constraints to propose
options; then use explicit feedback to reconsider. This is a testable design
analogy to flexibility, not a claim that an LLM or ranking model measures the
psychological construct. First validate the operationalization and avoid assuming
that greater switching is always better.

### Ryan and Deci (2000): self-determination theory

This foundational synthesis addresses autonomy, competence, relatedness, and
forms of motivation. Autonomy concerns volition and endorsement, not simply
having more buttons or acting without help. A reasoned suggestion can support
autonomy when the user can understand, modify, or decline it.

For Luna, explain why an option might fit, make alternatives manageable, and
allow rejection without pressure. Small feasible activities may support perceived
competence; relatedness should not be equated with an AI impersonating human
attachment. These product mappings are hypotheses, not direct findings from an
evaluation of conversational AI.

Ranking options measures relative preference; post-use ratings measure reported
experience. Neither directly establishes need satisfaction or clinical change.
If we claim autonomy support, use an appropriate validated measure and check its
applicability to the population and digital setting.

### Proposed integration and evaluation logic

Use regulatory flexibility as an organizing hypothesis, guided discovery as a
conversational method, optional affect labeling as a user-controlled aid,
behavioural-activation principles to inform meaningful action, and autonomy
support to govern choice. This combination has not been validated as a package.

Evaluate components separately before attributing benefit to the combined system.
Compare guided questions against the current reflection style while measuring
grounding, perceived insight, invalidation, and conversational burden. Compare
optional emotion suggestions against user-generated labels while checking
anchoring and corrections. Compare the fixed catalog with contextual selection
using preference, feasibility, uptake, and post-use helpfulness as separate
outcomes. More disclosure or longer sessions should not automatically count as
success. Before/after mood change alone is not a causal treatment effect.

Full-text reading, short verified quotations with page/section locations, recent
review updates, and direct conversational-agent evidence remain outstanding.
