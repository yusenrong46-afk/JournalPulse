# Emotion Journal Evaluation

- Generated: 2026-04-14T02:41:12.697604+00:00
- Selected production model: `transformer`
- Selected classical explainer: `tfidf-linearsvc`
- Logistic Regression macro F1: `0.7264`
- LinearSVC macro F1: `0.8224`
- Transformer macro F1: `0.86`

## Test Metrics

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Logistic Regression (tfidf-logreg) | 0.8295 | 0.7264 |
| LinearSVC (tfidf-linearsvc) | 0.8795 | 0.8224 |
| Transformer (distilroberta-base) | 0.9 | 0.86 |

## Example Predictions

- `correct` | true=`sadness` | predicted=`sadness` | confidence=`0.9549` | text="im feeling rather rotten so im not very ambitious right now"
- `correct` | true=`sadness` | predicted=`sadness` | confidence=`0.9914` | text="im updating my blog because i feel shitty"
- `incorrect` | true=`joy` | predicted=`fear` | confidence=`0.5344` | text="i explain why i clung to a relationship with a boy who was in many ways immature and uncommitted despite the excitement i should have been feeling for getting accepted into the masters program at the university of virginia"
- `incorrect` | true=`fear` | predicted=`anger` | confidence=`0.7615` | text="i don t feel particularly agitated"
- `incorrect` | true=`sadness` | predicted=`love` | confidence=`0.8764` | text="im not sure the feeling of loss will ever go away but it may dull to a sweet feeling of nostalgia at what i shared in this life with my dad and the luck i had to have a dad for years"
