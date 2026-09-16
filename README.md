# Hiver AI Support Agent

An AI-powered customer support agent built for the Hiver SDE Intern take-home assignment.

The system uses historical customer-support interactions from Twitter to:

1. Classify incoming customer messages into a small set of support intents.
2. Retrieve historically similar customer-support cases.
3. Draft a concise response grounded in those historical cases.
4. Decide whether the case can be auto-handled or should be escalated to a human.

The system is designed as a conservative support pipeline: automation is allowed only when the intent is sufficiently confident, relevant historical evidence is available, and the request does not trigger a human-review rule.

---

## 1. Problem Framing

Customer support messages are short, noisy, and often contain overlapping signals.

For example:

> My iPhone battery is draining really quickly after the latest update

contains both update-related and battery-related language.

A simple keyword-based system could classify this as a software-update problem even though the customer's primary problem is battery drain.

Therefore, the system uses a **primary-problem labeling rule**.

Examples:

| Customer message | Intent |
|---|---|
| iOS update won't install | `software_update` |
| iOS update ruined my battery | `battery_charging` |
| iOS update broke my Wi-Fi | `connectivity` |
| iOS update makes my phone freeze | `device_performance` |

The objective is not to maximize automation at all costs.

Instead, the system should:

- make a confident prediction when possible
- ground responses in historical support behavior
- avoid unsupported claims
- escalate uncertain or risky cases

---

# 2. System Architecture

```text
                         Customer Message
                                |
                                v
                    +-----------------------+
                    |   Intent Classifier   |
                    |     TF-IDF + SVM      |
                    +-----------------------+
                                |
                       Intent + Confidence
                                |
                                v
                    +-----------------------+
                    | Historical Retrieval  |
                    |    TF-IDF Similarity  |
                    +-----------------------+
                                |
                         Top Cases + Score
                                |
                                v
                    +-----------------------+
                    |  Escalation Policy    |
                    +-----------------------+
                         /              \
                        /                \
                       v                  v
                Human Escalation     Auto-handle
                                           |
                                           v
                                +-------------------+
                                |   Qwen2.5 3B      |
                                | Response Generator|
                                +-------------------+
                                           |
                                           v
                                +-------------------+
                                | Response Validator|
                                +-------------------+
                                           |
                                           v
                                   Final Response
3. Dataset

The project uses the Customer Support on Twitter dataset from Kaggle.

The full dataset contains approximately 2.8 million tweets covering customer interactions with many company support accounts.

The dataset contains both:

inbound customer messages
outbound support responses

After inspecting support-account activity, AppleSupport was selected as the target brand because it has a large number of support interactions.

The AppleSupport subset contains approximately 106k AppleSupport tweets.

Support Case Reconstruction

A Twitter conversation cannot always be treated as one support case.

Public support tweets can receive replies from multiple unrelated customers. Treating the entire public thread as a single conversation can therefore incorrectly merge unrelated issues.

Instead, the pipeline reconstructs direct:

Customer tweet → AppleSupport reply

pairs.

This produced approximately 106k direct customer-support pairs.

For the final lightweight retrieval artifact, a deduplicated sample of approximately 10k historical AppleSupport cases is stored in:

data/retrieval_apple_cases.csv
4. Brand Selection

AppleSupport was selected after comparing the volume of support activity across the available support accounts.

The final pipeline therefore focuses on:

Brand: Apple
Support account: AppleSupport

This makes the taxonomy, historical evidence, and response generation brand-specific rather than attempting to build a generic support agent across unrelated companies.

5. Intent Taxonomy

The final taxonomy contains 10 supported intents.

An additional other_unclear category was retained during analysis for messages that could not be confidently mapped to the supported taxonomy.

Intent	Description
battery_charging	Battery drain, charging and battery-life problems
keyboard_text_input	Keyboard, typing, autocorrect and text-input problems
connectivity	Wi-Fi, Bluetooth, cellular and connectivity problems
apple_id_account	Apple ID, login, account and authentication problems
messaging	iMessage, SMS and messaging problems
apple_app_service	Apple apps and services such as Music, App Store and related services
photos_media	Photos, camera and media-related problems
purchase_payment_order	Purchases, payments, subscriptions and order-related problems
software_update	Problems installing or completing software updates
device_performance	Freezing, crashes, lag and general device-performance problems
other_unclear	Messages that cannot be confidently mapped to the supported taxonomy
Intent Discovery

Semantic clustering was initially used to explore recurring themes in the AppleSupport data.

The clustering step was treated as intent discovery, not as the final source of labels.

The final taxonomy was manually defined from the recurring problem categories and validated against representative examples.

6. Golden Evaluation Set

A 200-example golden evaluation set was created from sampled AppleSupport customer messages.

Each example was reviewed and assigned to the final intent taxonomy.

The final distribution includes:

198 examples across supported intents
2 examples labeled other_unclear

The golden set is stored at:

data/golden/apple_golden_200_labeled.csv

The other_unclear examples are excluded from classifier accuracy calculations because they are not part of the supported intent taxonomy.

Annotation Limitation

An independent second human annotation pass was not completed.

Therefore, formal inter-annotator agreement such as Cohen's kappa is not reported.

This is an explicit limitation of the current evaluation rather than treating the single annotation pass as stronger evidence than it is.

7. Intent Classification

The classifier uses:

TF-IDF text features
Linear Support Vector Machine
class-balanced training
probability calibration for confidence estimation

The classifier was compared against multiple baselines.

Baseline Results
Model	Accuracy	Macro-F1	Weighted-F1
Majority	6.1%	1.1%	0.7%
Naive Bayes	58.6%	59.1%	60.4%
Logistic Regression	86.9%	86.6%	87.7%
Linear SVM	86.9%	86.7%	87.7%

The baseline results show that simple linear text classifiers substantially outperform the majority and Naive Bayes baselines on this evaluation set.

The Linear SVM was selected for the final classifier.

8. Hyperparameter Tuning

A grid search was performed over:

SVM regularization
class weighting
TF-IDF minimum document frequency
n-gram range
sublinear term frequency

The selected configuration was:

C = 0.5
class_weight = balanced
min_df = 1
ngram_range = (1, 1)
sublinear_tf = True

The best 5-fold cross-validation macro-F1 was:

0.8307

The tuning result is stored in:

data/processed/apple/classification/tuning_results.csv
9. Classifier Confidence

The final classifier is calibrated using sigmoid probability calibration.

The purpose of calibration is not necessarily to improve classification accuracy.

Instead, it provides a more useful confidence estimate for the downstream escalation policy.

The agent uses classifier confidence as one of the signals for deciding whether a case can be automatically handled.

10. Historical Case Retrieval

The response generator should not freely invent troubleshooting procedures, policies, refunds, guarantees, or other support information.

Instead, the system retrieves historically similar AppleSupport cases.

Each retrieved case contains:

Customer message
Support response
Similarity score

The final retrieval implementation uses:

TF-IDF + cosine similarity

rather than a heavyweight vector database or neural embedding dependency.

This design has two advantages:

It is lightweight and easy to reproduce locally.
It avoids requiring large native ML dependencies for the final runnable repository.

The retrieval source corpus is:

data/retrieval_apple_cases.csv

The serialized retrieval model is:

models/apple_support_retrieval.joblib
11. Why TF-IDF Retrieval?

An earlier prototype used semantic embeddings with FAISS.

However, the local environment had native-library compatibility restrictions that prevented reliable execution of the embedding/FAISS stack.

Rather than making the submission dependent on a fragile environment-specific setup, the final implementation uses TF-IDF retrieval.

This also keeps the architecture easy to inspect:

Customer message
       ↓
TF-IDF vector
       ↓
Cosine similarity against historical cases
       ↓
Top-k historical cases

The retrieval layer is therefore intentionally simple and reproducible.

12. Response Generation

Response generation uses a locally running:

Qwen2.5 3B

model through Ollama.

The LLM receives:

Customer message
Predicted intent
Retrieved historical customer/support cases

The prompt instructs the model to:

use historical cases as the primary source of information
avoid copying historical responses word-for-word
avoid inventing troubleshooting steps
avoid inventing policies
avoid inventing refunds or guarantees
avoid unsupported URLs
avoid usernames
avoid dataset identifiers
avoid introducing unsupported device/software versions
avoid destructive actions unless historically supported
keep responses concise
return only the customer-facing response

If the historical evidence is insufficient, the model is instructed to indicate that the case requires human assistance.

13. Escalation Policy

The agent does not automatically respond to every classified message.

A case is escalated when one or more of the following conditions are met.

Rule 1: Unsupported Intent

If the predicted intent is:

other_unclear

the case is escalated.

Reason:

The message could not be mapped to a supported intent.
Rule 2: Low Classifier Confidence

If:

classifier confidence < 0.80

the case is escalated.

Reason:

Classifier confidence is below the safe handling threshold.
Rule 3: Weak Historical Evidence

If no historical cases are retrieved, the case is escalated.

If the top retrieved case has:

similarity < 0.35

the case is also escalated.

The retrieval threshold is specific to the final TF-IDF implementation.

The previous 0.70 threshold belonged to the earlier neural embedding retrieval prototype and is not used by the final implementation.

Rule 4: Risky Requests

Certain requests are explicitly routed to human review.

Examples include:

refund requests
chargebacks
duplicate charges
subscription cancellation
account deletion
unauthorized purchases
legal escalation
stolen devices
hacking/security-related requests
destructive device actions

The purpose is to avoid allowing a language model to independently make sensitive account, financial, legal, or destructive decisions.

14. Response Validation

Even after the escalation policy allows automatic handling, the generated response passes through a deterministic validator.

The validator checks for:

Empty responses

A response must contain actual customer-facing content.

Username leakage

The response must not contain Twitter usernames such as:

@username
URL leakage

Historical URLs are blocked because they may no longer be valid or appropriate.

Unsupported version information

The response should not introduce a specific version such as:

iOS 11.1.1

unless that version was explicitly provided by the customer.

If validation fails, the response is discarded and the case is escalated to a human.

15. End-to-End Agent Flow

For every incoming message:

1. Receive customer message
             ↓
2. Predict support intent
             ↓
3. Estimate classifier confidence
             ↓
4. Retrieve similar historical cases
             ↓
5. Check retrieval similarity
             ↓
6. Check risk rules
             ↓
7. If unsafe/uncertain → Human escalation
             ↓
8. Otherwise generate response
             ↓
9. Validate generated response
             ↓
10. If validation fails → Human escalation
             ↓
11. Otherwise return response
16. Example

Input:

My iPhone battery is draining really quickly after the latest update

The classifier predicts:

Intent:
battery_charging

with a high classifier confidence.

The historical retrieval layer searches for previous AppleSupport cases involving similar battery problems.

The escalation layer then checks:

Intent confidence
+
Historical similarity
+
Risk triggers

If all checks pass, Qwen generates a concise response grounded in the retrieved historical cases.

If any safety or evidence check fails, the case is escalated instead.

17. Evaluation

The project evaluates different parts of the system separately because classification accuracy alone does not measure the quality of a support agent.

The evaluation areas are:

Intent Classification
        +
Historical Retrieval
        +
Response Quality
        +
Escalation
        +
Response Safety
17.1 Classification Evaluation

The Linear SVM achieved:

Accuracy:     86.9%
Macro-F1:     86.7%
Weighted-F1:  87.7%

on the 198 supported-intent examples in the golden evaluation set.

The best tuned configuration achieved:

5-fold CV Macro-F1: 83.1%

The baseline results are stored in:

data/processed/apple/classification/baseline_results.csv
18. Classification Failure Analysis

The classifier produced 26 misclassified examples in the evaluated golden set.

The major failure patterns were:

Failure Mode 1: Update-Context Bias

Messages mentioning iOS updates were sometimes classified as:

software_update

even when the primary problem was:

battery
connectivity
device performance
account problems
Hypothesis

The training data contains strong lexical associations between update-related vocabulary and the software_update class.

Mitigation

Use the primary-problem labeling rule so that the underlying customer problem takes priority over contextual words such as "update".

Failure Mode 2: Connectivity vs Apple Service

Some messages about connectivity to Apple services were difficult to distinguish from general connectivity problems.

Hypothesis

Words such as:

connect
internet
working
service

can occur across multiple intents.

Possible improvement

Use hierarchical classification:

Connectivity
      ↓
Wi-Fi / Cellular / Bluetooth / Apple Service
Failure Mode 3: Payment/Subscription vs Apple Service

Some purchase, payment, and subscription issues overlap strongly with App Store and Apple service messages.

Hypothesis

The same vocabulary can describe both the service and the transaction involving that service.

Possible improvement

Use explicit entity/action features such as:

payment
charged
subscription
purchase
refund
order

before determining the final intent.

Failure Mode 4: Multi-Intent Messages

Some customer messages contain multiple problems.

For example:

After updating my phone my Wi-Fi stopped working and my battery is also draining.

A single-label classifier must choose one primary intent.

Possible improvement

Use multi-label classification internally and select a primary intent only for routing.

Failure Mode 5: Sparse or Ambiguous Messages

Very short messages can lack enough information to determine the actual issue.

Examples can contain only:

Yes
6S
11.0.3
Still not working

These messages may be meaningful in the original conversation context but ambiguous when evaluated independently.

Possible improvement

Use conversation context when available rather than relying only on a single tweet.

19. Response Evaluation

A prototype response evaluation was performed using a local LLM judge.

The generated responses were evaluated on:

relevance
groundedness
helpfulness
safety

The evaluation was performed on a sample of generated responses.

Because both generation and judging involved a local language model, the results should be interpreted as an early quality signal rather than definitive human evaluation.

A stronger evaluation would use independent human reviewers.

20. What Is Misleading About My Headline Number?

A single headline number such as:

86.9% classification accuracy

does not represent the complete performance of the support agent.

There are several reasons.

20.1 The Evaluation Set Is Small

The classifier evaluation uses 198 supported-intent golden examples.

That is useful for detecting major problems, but it is not large enough to establish production-level reliability.

20.2 Golden Data Is Not Fully Independent

The golden examples were sampled from the same broader dataset/candidate-generation process used during development.

The exact golden IDs were excluded from weakly labelled training data, but the broader source distribution is still shared.

Therefore, the evaluation should not be interpreted as equivalent to a completely independent production test set.

20.3 The Dataset Is Temporally Concentrated

AppleSupport activity in the dataset is heavily concentrated in 2017.

This means a random train/test split can contain very similar language distributions across training and evaluation.

A chronological evaluation would provide stronger evidence about performance on future support messages.

20.4 Classification Accuracy Is Not Resolution Success

Predicting:

battery_charging

correctly does not prove that the generated response solves the customer's battery problem.

The complete system requires separate measurement of:

intent correctness
retrieval relevance
response groundedness
response helpfulness
escalation correctness
actual resolution success
20.5 Retrieval Similarity Is Not Evidence Quality

A high text similarity score does not guarantee that the retrieved historical case contains the correct resolution.

Two messages can use similar words while requiring different actions.

Human evaluation of retrieved evidence would therefore be more meaningful.

20.6 Automation Rate Is Not Resolution Rate

If a case is auto-handled, that means the routing policy allowed the system to generate a response.

It does not prove that the customer issue was successfully resolved.

Therefore:

Auto-handled ≠ Resolved

This distinction is important for any production support system.

21. One More Week

With one additional week, I would prioritize the following improvements.

21.1 Independent Human Evaluation

Create a larger golden dataset with at least two independent annotators.

Measure:

intent agreement
Cohen's kappa
retrieval relevance
response helpfulness
response groundedness
escalation correctness
21.2 Temporal Evaluation

Create a chronological evaluation split.

For example:

Earlier data → Training
Later data  → Evaluation

This would better approximate deployment on future customer messages.

21.3 Hierarchical Intent Classification

Instead of directly predicting all intents, use:

Broad Category
       ↓
Specific Intent

This could help distinguish related categories such as:

Connectivity
    ├── Wi-Fi
    ├── Cellular
    └── Bluetooth

Transactions
    ├── Purchase
    ├── Payment
    ├── Subscription
    └── Refund
21.4 Human Retrieval Evaluation

Have reviewers judge whether the top retrieved historical cases are:

Relevant
Partially Relevant
Irrelevant

This would provide a better evaluation of retrieval quality than cosine similarity alone.

21.5 Structured Historical Evidence

Instead of passing raw historical responses directly to the LLM, extract structured resolution patterns:

Problem
Relevant Context
Troubleshooting Action
Escalation Condition
Resolution

This would make response generation more controllable.

21.6 Better Response Safety

Add deterministic checks for:

unsupported troubleshooting claims
financial actions
account actions
destructive device actions
policy claims
unsupported URLs
unsupported software versions
21.7 Cost-Sensitive Escalation

The escalation threshold should ultimately be selected based on the business cost of different errors.

For example:

Wrong automated answer
        vs
Unnecessary human escalation

These two errors have different operational costs.

A production system should optimize the threshold using those costs rather than selecting a threshold only from model statistics.

22. Decision Log

The following decisions shaped the final implementation.

#	Decision	Reason
1	Select AppleSupport	Large volume of historical support interactions
2	Use direct customer→support pairs	Avoid incorrectly merging unrelated public Twitter replies
3	Use clustering only for discovery	Semantic clusters showed overlapping intent boundaries
4	Define explicit support taxonomy	Needed consistent labels for supervised classification
5	Use primary-problem labeling	Prevent update vocabulary from dominating unrelated issues
6	Retain other_unclear	Avoid forcing ambiguous messages into incorrect classes
7	Compare multiple classifier baselines	Establish a meaningful baseline before tuning
8	Select Linear SVM	Strong performance with lightweight deployment
9	Tune TF-IDF/SVM parameters	Improve classification performance systematically
10	Calibrate classifier probabilities	Provide confidence estimates for escalation
11	Retrieve historical support cases	Ground generated responses in observed support behavior
12	Replace FAISS/embedding retrieval with TF-IDF	Improve local reproducibility and avoid native dependency issues
13	Add confidence and similarity gates	Avoid automatically handling uncertain cases
14	Add risky-request rules	Route sensitive financial/account/legal/security cases to humans
15	Add response validation	Prevent usernames, URLs and unsupported version information from leaking into responses
23. Repository Structure
HIVER-AGENT/
│
├── app/
├── config/
│
├── data/
│   ├── golden/
│   │   └── apple_golden_200_labeled.csv
│   │
│   ├── processed/
│   │   └── apple/
│   │       └── classification/
│   │
│   ├── raw/
│   │   └── sample.csv
│   │
│   └── retrieval_apple_cases.csv
│
├── experiments/
│
├── models/
│   ├── apple_intent_classifier.joblib
│   └── apple_support_retrieval.joblib
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── data/
│   │   ├── dataset_inspection.py
│   │   ├── brand_analysis.py
│   │   ├── brand_filter.py
│   │   ├── support_pairs.py
│   │   ├── analyze_customer_messages.py
│   │   ├── clean_pairs.py
│   │   ├── intent_exploration.py
│   │   ├── intent_clustering.py
│   │   ├── intent_discovery.py
│   │   ├── intent_validation.py
│   │   ├── semantic_intent_validation.py
│   │   ├── temporal_analysis.py
│   │   ├── text_cleaning.py
│   │   └── create_golden_set.py
│   │
│   ├── ml/
│   │   ├── prepare_data.py
│   │   ├── build_training_set.py
│   │   ├── train_baselines.py
│   │   ├── analyze_classifier.py
│   │   ├── tune_classifier.py
│   │   └── calibrate_classifier.py
│   │
│   ├── retrieval/
│   │   ├── build_index.py
│   │   └── search.py
│   │
│   └── agent/
│       ├── generate_response.py
│       ├── run_agent.py
│       ├── escalation.py
│       ├── evaluate_escalation.py
│       ├── evaluate_agent.py
│       ├── response_validator.py
│       └── evaluate_responses.py
│
├── tests/
│   └── test_conversation.py
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
24. Setup
Requirements
Python 3.11+
Ollama
Qwen2.5 3B model

The Python dependencies are listed in:

requirements.txt
Step 1: Clone the repository
git clone <repository-url>
cd HIVER-AGENT
Step 2: Create a virtual environment
Windows
python -m venv venv
venv\Scripts\activate
Step 3: Install dependencies
pip install -r requirements.txt
25. Configure Ollama

Install Ollama and download the local model:

ollama pull qwen2.5:3b

Make sure the Ollama service is running.

Create a .env file in the project root:

LLM_BACKEND=ollama
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen2.5:3b

The .env file should not be committed to Git.

A template is provided as:

.env.example
26. Run the Agent

Run the agent from the project root:

python -m src.agent.run_agent

The program asks for a customer message.

Example:

Customer message: My iPhone battery is draining really quickly after the latest update

The agent then displays:

Customer message
Predicted intent
Classifier confidence
Escalation decision
Escalation reason
Generated response

when automatic handling is allowed.

27. Run Tests

Run:

pytest

The tests verify core project functionality.

28. Reproducibility

The repository includes the trained classifier and retrieval artifact so that the main agent can be run without retraining the models.

The final runtime path is therefore:

Install dependencies
        ↓
Configure Ollama
        ↓
Run src.agent.run_agent
        ↓
Enter customer message
        ↓
Receive classification + routing + response

The large raw Twitter dataset is intentionally not committed to the repository.

29. Limitations
Historical Data

The source dataset ends in 2017.

Modern Apple products, software versions, policies, and support workflows may differ significantly.

Twitter-Specific Language

The dataset consists of short Twitter messages.

These messages may contain:

abbreviations
incomplete context
typos
mentions
URLs
short follow-up messages

A production support system would generally have access to richer conversation context.

Limited Golden Set

The current evaluation set contains only 200 reviewed examples.

A much larger independently annotated evaluation set would provide stronger evidence.

No Independent Inter-Annotator Agreement

A second independent human annotation pass was not completed.

Therefore, formal annotation agreement is not reported.

LLM-as-Judge Limitation

The response evaluation uses an LLM judge.

LLM-based evaluation can provide useful signals but should not replace independent human review for a customer-support system.

Retrieval Limitation

TF-IDF similarity is based on lexical overlap.

It can fail when two messages have the same meaning but use substantially different wording.

A stronger production system could use a hybrid retrieval approach combining:

Lexical Retrieval
+
Semantic Retrieval
+
Intent Filtering
Resolution Limitation

The system evaluates whether a response can be generated safely.

It does not measure whether the customer's actual issue was successfully resolved.

30. Security and Safety Considerations

The system intentionally avoids autonomous handling of certain high-risk requests.

Human escalation is preferred for:

financial disputes
unauthorized purchases
account deletion
legal threats
stolen devices
security compromises
destructive device operations

The response validator also prevents common forms of historical-data leakage.

This follows a simple principle:

When evidence or safety is uncertain,
escalate rather than hallucinate.
31. Final Design Principle

The core design principle of the system is:

Automation should be gated by evidence rather than driven solely by model confidence.

The complete pipeline combines:

Intent Classification
        +
Historical Retrieval
        +
Confidence Threshold
        +
Similarity Threshold
        +
Risk Rules
        +
Response Validation