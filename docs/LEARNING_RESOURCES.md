# JournalPulse Learning Resources

Last reviewed: 2026-04-23

This project combines software engineering, machine learning, NLP, deep learning,
web APIs, UI design, databases, testing, and responsible AI. Use this document as
a map: learn a concept, then open the related project files while the idea is
fresh.

## How to Use This Guide

Do not try to study everything at once. The fastest path is:

1. Understand the product flow in `app/streamlit/app.py` and `src/emotion_journal/experience.py`.
2. Learn classical text classification: preprocessing, TF-IDF, Logistic Regression, LinearSVC.
3. Learn deep learning inference: tokenization, logits, softmax, transformer sequence classification.
4. Learn evaluation: accuracy, precision, recall, F1, macro F1, confusion matrix.
5. Learn the engineering shell: FastAPI, Pydantic, SQLite, Streamlit, pytest.
6. Learn responsible AI: safety boundaries, model cards, privacy, mental-health disclaimers.

## Project Map

| Area | Project Files |
| --- | --- |
| Streamlit UI | `app/streamlit/app.py` |
| Product orchestration | `src/emotion_journal/experience.py` |
| Model loading and prediction | `src/emotion_journal/model.py` |
| Text cleanup and crisis keyword scan | `src/emotion_journal/preprocessing.py` |
| Recommendation templates | `src/emotion_journal/recommendations.py` |
| Curated resource ranking | `src/emotion_journal/resources.py` |
| Guided coach | `src/emotion_journal/coach.py`, `src/emotion_journal/llm.py` |
| API backend | `src/emotion_journal/api.py`, `src/emotion_journal/schemas.py` |
| Database | `src/emotion_journal/db.py` |
| Analytics | `src/emotion_journal/analytics.py` |
| Training pipeline | `scripts/train.py` |
| Tests | `tests/` |

## 1. Python and Code Structure

Why it matters: almost every file in this project uses functions, imports,
dataclasses, type hints, paths, dictionaries, and lists.

Start with:

- Official Python Tutorial: https://docs.python.org/3/tutorial/
- Python standard library reference: https://docs.python.org/3/library/
- Python `dataclasses`: https://docs.python.org/3/library/dataclasses.html
- Python `pathlib`: https://docs.python.org/3/library/pathlib.html
- Python type hints: https://docs.python.org/3/library/typing.html

Practice in this repo:

- Read `Prediction` in `src/emotion_journal/model.py`.
- Read `build_prediction_experience()` in `src/emotion_journal/experience.py`.
- Read path constants in `src/emotion_journal/config.py`.

## 2. Data Handling with pandas and NumPy

Why it matters: the training script converts Hugging Face dataset splits into
pandas dataframes, evaluates predictions with NumPy arrays, and writes metrics.

Learn:

- pandas getting started: https://pandas.pydata.org/docs/getting_started/index.html
- pandas user guide: https://pandas.pydata.org/docs/user_guide/index.html
- NumPy absolute beginner guide: https://numpy.org/doc/stable/user/absolute_beginners.html
- NumPy basics: https://numpy.org/doc/stable/user/basics.html

Practice in this repo:

- Open `scripts/train.py`.
- Find `load_training_splits()`.
- Find `evaluate_predictions()`.
- Notice where `np.argmax`, `np.asarray`, and dataframe columns are used.

## 3. Natural Language Processing Foundations

Why it matters: JournalPulse is a text classification app. The input is raw
human language, so you need to understand tokenization, normalization, word
features, embeddings, and sequence classification.

Best textbook:

- Jurafsky and Martin, Speech and Language Processing: https://web.stanford.edu/~jurafsky/slp3/

Best university course:

- Stanford CS224N, Natural Language Processing with Deep Learning: https://web.stanford.edu/class/cs224n/

Video intuition:

- Stanford CS224N lecture videos are linked from the course page: https://web.stanford.edu/class/cs224n/
- 3Blue1Brown neural networks and transformers: https://www.3blue1brown.com/topics/neural-networks

Practice in this repo:

- Read `normalize_text()` in `src/emotion_journal/preprocessing.py`.
- Read the tokenizer call in `ArtifactPredictor._predict_probabilities()` in `src/emotion_journal/model.py`.

## 4. Text Preprocessing

Why it matters: the project standardizes text before training classical models
and before explanation. That makes phrase extraction and crisis keyword matching
more predictable.

Learn:

- scikit-learn text feature extraction: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- Python regular expressions: https://docs.python.org/3/library/re.html
- Python string methods: https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str

Practice in this repo:

- Open `src/emotion_journal/preprocessing.py`.
- Understand each step: lowercase, URL removal, punctuation removal, whitespace cleanup.
- Then open `tests/test_core.py` and read `test_normalize_text_removes_punctuation_and_urls()`.

## 5. Classical Machine Learning: TF-IDF, Logistic Regression, LinearSVC

Why it matters: the project trains classical baselines and keeps the best one as
the phrase-level explainer.

Core docs:

- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
- `TfidfVectorizer`: https://sklearn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Logistic Regression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Support Vector Machines: https://scikit-learn.org/stable/modules/svm.html
- `Pipeline`: https://scikit-learn.org/stable/modules/compose.html#pipeline

Videos:

- StatQuest video index, especially Logistic Regression, Machine Learning, and Neural Networks: https://statquest.org/video_index.html

Book:

- An Introduction to Statistical Learning, Python edition: https://www.statlearning.com/

Practice in this repo:

- Open `scripts/train.py`.
- Read `train_logistic_baseline()`.
- Read `train_linear_svc_baseline()`.
- Open `src/emotion_journal/model.py` and read `BaselineExplainer`.

Key idea:

```text
TF-IDF turns text into numbers.
Logistic Regression / LinearSVC learn weights for each phrase.
The explainer uses those weights to show which phrases pushed toward the predicted emotion.
```

## 6. Model Evaluation

Why it matters: this project does not simply train a model and hope. It compares
models using test metrics and promotes the best production model by macro F1.

Learn:

- scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- `classification_report`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- `confusion_matrix`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- `f1_score`: https://sklearn.org/stable/modules/generated/sklearn.metrics.f1_score.html

Good video source:

- StatQuest video index, especially precision, recall, F1, ROC/AUC, confusion matrix: https://statquest.org/video_index.html

Practice in this repo:

- Open `scripts/train.py`.
- Read `evaluate_predictions()`.
- Open `artifacts/reports/evaluation.md`.
- Open `artifacts/models/production.json`.

Key idea:

```text
Accuracy asks: how often was the model right?
Macro F1 asks: how well did the model do across all classes, including smaller ones?
```

Macro F1 matters here because emotions like `love` and `surprise` have fewer
examples than `joy` and `sadness`.

## 7. Neural Networks and Deep Learning Foundations

Why it matters: the selected production model is a fine-tuned transformer, and
transformers are deep neural networks.

Beginner intuition:

- 3Blue1Brown neural networks: https://www.3blue1brown.com/topics/neural-networks
- StatQuest neural networks and deep learning videos: https://statquest.org/video_index.html

Practical textbook:

- Dive into Deep Learning: https://d2l.ai/

Theory reference:

- Goodfellow, Bengio, and Courville, Deep Learning: https://www.deeplearningbook.org/

Practice in this repo:

- Open `scripts/train.py`.
- Find the `EmotionDataset` class inside `train_transformer()`.
- Find the training loop that calls `outputs.loss.backward()`.
- Find where the best validation model state is saved.

## 8. PyTorch

Why it matters: PyTorch handles the transformer training and inference. You need
to understand tensors, modules, devices, gradients, optimizers, and evaluation
mode.

Official learning:

- PyTorch tutorials: https://docs.pytorch.org/tutorials/
- PyTorch basics: https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- Deep Learning for NLP with PyTorch: https://docs.pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html
- PyTorch paper: https://arxiv.org/abs/1912.01703

Practice in this repo:

- Open `scripts/train.py`.
- Search for `torch.device`.
- Search for `DataLoader`.
- Search for `torch.no_grad()`.
- Search for `torch.softmax`.
- Open `src/emotion_journal/model.py` and read the transformer inference branch.

Key idea:

```text
Training mode learns weights.
Evaluation mode uses learned weights without updating them.
torch.no_grad() saves memory and prevents gradient tracking during prediction.
```

## 9. Transformers and Hugging Face

Why it matters: JournalPulse uses a Hugging Face transformer model for production
emotion classification.

Best learning path:

- Hugging Face Course: https://huggingface.co/course
- Hugging Face Transformers text classification guide: https://huggingface.co/docs/transformers/tasks/sequence_classification
- Hugging Face Transformers docs: https://huggingface.co/docs/transformers
- Hugging Face Datasets docs: https://huggingface.co/docs/datasets
- Hugging Face paper: https://arxiv.org/abs/1910.03771

Deep NLP course:

- Stanford CS224N: https://web.stanford.edu/class/cs224n/

Visual intuition:

- 3Blue1Brown transformers and attention lessons: https://www.3blue1brown.com/topics/neural-networks

Practice in this repo:

- Open `scripts/train.py`.
- Find `AutoTokenizer.from_pretrained(...)`.
- Find `AutoModelForSequenceClassification.from_pretrained(...)`.
- Open `src/emotion_journal/model.py`.
- Find where tokenization happens during prediction.

Key idea:

```text
Tokenizer: text -> token IDs
Transformer: token IDs -> logits
Softmax: logits -> probabilities
Argmax: probabilities -> predicted emotion
```

## 10. Fine-Tuning and Transfer Learning

Why it matters: the app does not train a language model from scratch. It starts
from a pretrained transformer and fine-tunes it for six emotion labels.

Learn:

- Hugging Face text classification fine-tuning: https://huggingface.co/docs/transformers/tasks/sequence_classification
- Hugging Face Course, fine-tuning section: https://huggingface.co/course
- Stanford CS224N transformer lectures: https://web.stanford.edu/class/cs224n/

Practice in this repo:

- Open `scripts/train.py`.
- Find where the transformer is loaded.
- Find where lower layers are frozen.
- Find where validation macro F1 chooses the best checkpoint.

Key idea:

```text
Pretraining gives the model general language understanding.
Fine-tuning teaches the model this project's specific labels.
```

## 11. Explainability

Why it matters: users should not only see `joy 91%`; they should also see clues
about what language the system noticed.

Learn:

- scikit-learn text features: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- scikit-learn linear models: https://scikit-learn.org/stable/modules/linear_model.html
- Model Cards for Model Reporting: https://research.google/pubs/pub48120/

Practice in this repo:

- Open `src/emotion_journal/model.py`.
- Read `BaselineExplainer`.
- Notice this idea:

```text
phrase contribution = TF-IDF value * class weight
```

This is why the classical model remains useful even though the transformer is
the production predictor.

## 12. Recommendation Systems and Ranking

Why it matters: after predicting an emotion, the app recommends curated support
resources. That ranking is simple, readable, and easy to improve.

Learn:

- scikit-learn nearest neighbors and similarity concepts: https://scikit-learn.org/stable/modules/neighbors.html
- Recommender systems overview from Google Developers: https://developers.google.com/machine-learning/recommendation

Practice in this repo:

- Open `src/emotion_journal/resources.py`.
- Read `_resource_matches()`.
- Read `_score_resource()`.
- Open `assets/resources/catalog.json`.

Key idea:

```text
Resources are content-filtered by emotion tags and coping style.
User feedback adjusts ranking:
helpful -> higher score
opened -> slightly higher score
dismissed -> lower score
```

## 13. FastAPI Backend

Why it matters: the project is not only a UI. It also exposes a real API with
prediction, entries, analytics, resources, and coach endpoints.

Learn:

- FastAPI docs: https://fastapi.tiangolo.com/
- FastAPI tutorial: https://fastapi.tiangolo.com/tutorial/
- FastAPI testing: https://fastapi.tiangolo.com/tutorial/testing/
- Uvicorn docs: https://www.uvicorn.org/

Practice in this repo:

- Open `src/emotion_journal/api.py`.
- Find `POST /predict`.
- Find `POST /entries`.
- Find `GET /analytics`.
- Run the API locally with:

```bash
PYTHONPATH=src uvicorn emotion_journal.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## 14. Pydantic Schemas and Validation

Why it matters: API inputs must be validated. Empty journal text should not reach
the model.

Learn:

- Pydantic models: https://docs.pydantic.dev/latest/concepts/models/
- Pydantic validators: https://docs.pydantic.dev/latest/concepts/validators/

Practice in this repo:

- Open `src/emotion_journal/schemas.py`.
- Read `JournalInput`.
- Read `clean_text()`.
- Read `PredictionResponse`.

Key idea:

```text
Pydantic turns untrusted JSON into validated Python objects.
```

## 15. Streamlit UI

Why it matters: Streamlit is the user-facing app. It turns your model and API
logic into something a recruiter can try.

Learn:

- Streamlit docs: https://docs.streamlit.io/
- Streamlit create an app tutorial: https://docs.streamlit.io/get-started/tutorials/create-an-app
- Streamlit tutorials: https://docs.streamlit.io/develop/tutorials
- Streamlit session state: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
- Streamlit caching: https://docs.streamlit.io/develop/concepts/architecture/caching

Practice in this repo:

- Open `app/streamlit/app.py`.
- Find `render_new_entry_page()`.
- Find `render_prediction_summary()`.
- Find `render_history_page()`.
- Find `render_insights_page()`.

Run:

```bash
PYTHONPATH=src streamlit run app/streamlit/app.py
```

## 16. SQLite and Persistence

Why it matters: saved entries, feedback, and resource interactions live in a
local SQLite database.

Learn:

- SQLite documentation: https://sqlite.org/docs.html
- Python `sqlite3`: https://docs.python.org/3/library/sqlite3.html
- SQL tutorial from SQLite: https://sqlite.org/lang.html

Practice in this repo:

- Open `src/emotion_journal/db.py`.
- Read `initialize_database()`.
- Read `insert_entry()`.
- Read `list_entries()`.
- Read `record_resource_interaction()`.

Key idea:

```text
The model predicts one entry.
The database gives the app memory over time.
```

## 17. Testing with pytest

Why it matters: tests protect the project while you refactor or improve it.

Learn:

- pytest docs: https://docs.pytest.org/en/stable/
- pytest getting started: https://docs.pytest.org/en/stable/getting-started.html
- FastAPI testing: https://fastapi.tiangolo.com/tutorial/testing/

Practice in this repo:

- Open `tests/test_core.py`.
- Open `tests/test_api.py`.
- Open `tests/test_model_smoke.py`.
- Run:

```bash
PYTHONPATH=src pytest
```

Key idea:

```text
Unit tests check small functions.
API tests check endpoint behavior.
Smoke tests check that the saved model artifacts load and run.
```

## 18. CI/CD and GitHub Portfolio Polish

Why it matters: a resume project looks much stronger when tests run
automatically.

Learn:

- GitHub Actions for Python: https://docs.github.com/en/actions/tutorials/build-and-test-code/python
- GitHub Actions docs: https://docs.github.com/en/actions
- Python packaging user guide: https://packaging.python.org/en/latest/

Practice next:

- Add `.github/workflows/test.yml`.
- Make it install `requirements.txt`.
- Make it run `PYTHONPATH=src pytest`.
- Add a status badge to `README.md`.

## 19. Model Cards, Dataset Cards, and Responsible AI

Why it matters: this project is wellness-adjacent. You need to be very clear
about intended use, limitations, safety behavior, and what the app is not.

Learn:

- Model Cards for Model Reporting: https://research.google/pubs/pub48120/
- Hugging Face model cards: https://huggingface.co/docs/hub/model-cards
- Hugging Face model card course section: https://huggingface.co/docs/course/chapter4/4
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
- NIST AI RMF 1.0 publication: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10

Practice in this repo:

- Open `artifacts/models/production.json`.
- Open `artifacts/reports/evaluation.md`.
- Create or improve `MODEL_CARD.md`.

Suggested sections:

- Intended use
- Out-of-scope use
- Dataset
- Labels
- Metrics
- Known limitations
- Safety behavior
- Privacy notes
- Ethical considerations

## 20. Mental Health Safety and Product Boundaries

Why it matters: JournalPulse should be positioned as reflective journaling
support, not therapy, diagnosis, or crisis counseling.

Learn from primary sources:

- 988 Lifeline official site: https://988lifeline.org/
- SAMHSA 988 page: https://www.samhsa.gov/mental-health/988
- WHO digital health and AI resources: https://www.who.int/health-topics/digital-health/artificial-intelligence
- WHO safe and ethical AI for health statement: https://www.who.int/news/item/16-05-2023-who-calls-for-safe-and-ethical-ai-for-health

Practice in this repo:

- Open `src/emotion_journal/config.py`.
- Find `DEFAULT_DISCLAIMER`.
- Find `CRISIS_SUPPORT_MESSAGE`.
- Find `CRISIS_KEYWORDS`.
- Open `src/emotion_journal/recommendations.py`.
- Read the crisis override in `build_support_response()`.

Key idea:

```text
The app should not pretend to be a clinician.
It can classify emotion, encourage reflection, and point to immediate human support when risk language appears.
```

## 21. Privacy and Local-First Thinking

Why it matters: journal text can be sensitive. A polished portfolio project
should explain where data is stored and how it could be protected.

Learn:

- OWASP Top 10: https://owasp.org/www-project-top-ten/
- OWASP API Security Top 10: https://owasp.org/API-Security/
- NIST Privacy Framework: https://www.nist.gov/privacy-framework
- APA digital mental health resources: https://www.psychiatry.org/psychiatrists/practice/mental-health-apps

Practice next:

- Add a privacy note to `README.md`.
- Add a demo mode with sample entries.
- Add a simple redaction helper for emails, phone numbers, and names before export.

## 22. Suggested Study Plan

### Week 1: Understand the running app

- Run the Streamlit app.
- Read `app/streamlit/app.py`.
- Read `src/emotion_journal/experience.py`.
- Learn Streamlit basics.
- Learn Python dataclasses and type hints.

### Week 2: Understand classical ML

- Study TF-IDF.
- Study Logistic Regression.
- Study LinearSVC.
- Read `scripts/train.py`.
- Read `BaselineExplainer` in `src/emotion_journal/model.py`.

### Week 3: Understand transformer inference

- Study tokenization.
- Study logits and softmax.
- Study Hugging Face sequence classification.
- Read `ArtifactPredictor._predict_probabilities()`.

### Week 4: Understand training and evaluation

- Study accuracy, precision, recall, F1, macro F1.
- Read `evaluate_predictions()`.
- Read the saved evaluation report.
- Explain why macro F1 is better than accuracy alone for this project.

### Week 5: Understand backend and storage

- Study FastAPI.
- Study Pydantic.
- Study SQLite.
- Read `api.py`, `schemas.py`, and `db.py`.

### Week 6: Make it portfolio-ready

- Add or improve `MODEL_CARD.md`.
- Add GitHub Actions.
- Add screenshots and a live demo.
- Add a clear privacy and safety section to the README.

## 23. Interview Explanation Checklist

You should be able to explain these without reading notes:

- What problem JournalPulse solves.
- Why this is text classification.
- What labels the model predicts.
- What TF-IDF means.
- Why Logistic Regression and LinearSVC are useful baselines.
- Why the transformer is stronger for production.
- What tokenization does.
- What logits are.
- What softmax does.
- Why macro F1 matters.
- How phrase-level explanation works.
- How crisis language changes the response.
- How resources are ranked.
- What SQLite stores.
- What FastAPI exposes.
- What Streamlit displays.
- What the project is not allowed to claim medically.

## 24. Best "If You Only Have Time for Five" Resources

1. Hugging Face Course: https://huggingface.co/course
2. Stanford CS224N: https://web.stanford.edu/class/cs224n/
3. scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
4. Dive into Deep Learning: https://d2l.ai/
5. FastAPI Tutorial: https://fastapi.tiangolo.com/tutorial/

