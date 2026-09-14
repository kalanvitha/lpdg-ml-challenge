Design Decisions — LPDG Innovation Hub Selection Challenge 2026
1. Part 2 Area Selection — Machine Learning
Choice
I selected Part 2 — Area E: Machine Learning.

Why I chose it
The challenge is about ranking gateways that should receive field attention. A machine-learning approach can learn patterns of unusual gateway behavior from historical telemetry and rank gateways according to their anomaly level.

I chose Machine Learning because it provides a way to move beyond a fixed statistical threshold while keeping the solution relatively simple, explainable, and practical for the live challenge.

Alternative considered
A more complex deep-learning or neural-network approach.

Why I rejected it
A deep-learning solution would add unnecessary complexity, require more tuning, and make the solution harder to explain and reproduce. The challenge emphasizes fewer clever parts, so I preferred a simpler anomaly-detection approach.

2. Input Features — Three Reliability Metrics
Choice
The final model uses these three telemetry metrics:

offline_duration_sec
disconnection_cnt
reboot_cnt
These metrics directly represent gateway reliability problems and are also used by the supplied 3-Sigma baseline.

Why I chose them
Using the same core reliability signals makes the comparison with the official baseline meaningful.

The three metrics also provide a simple and interpretable input to the ML model without unnecessarily increasing the feature space.

Alternatives considered
Using many more telemetry columns.

Using field-visit information.

Using engineer-review information as training labels.

Using meter-read information.

Why I rejected them
Using many additional telemetry columns could increase complexity and make the model harder to explain and maintain.

Field visits and engineer reviews are outcomes or supporting information rather than reliable official training labels. The hidden challenge ground truth is not available during development, so I did not train directly on proxy labels.

Meter-read information was not necessary for the core gateway-reliability ranking problem.

3. Gateway-Relative Features + Isolation Forest
Choice
I used Isolation Forest together with gateway-relative historical features.

For each gateway and each selected metric, the model creates features including:

raw metric value;

deviation from historical mean;

z-score;

ratio to historical mean;

deviation from historical maximum;

deviation from historical median.

Why I chose it
Different gateways can naturally have different operating ranges.

A value that is normal for one gateway may be unusual for another. Gateway-relative features allow the model to identify behavior that is unusual compared with the gateway's own history.

Isolation Forest was selected because the official hidden labels are unavailable for training. It can identify unusual observations without requiring supervised labels.

Alternatives considered
Alternative 1: Raw three-metric Isolation Forest

This would use only the raw telemetry values.

Alternative 2: A supervised classification model

For example, a decision tree or random forest classifier trained using engineer-review labels.

Why I rejected them
The raw-value approach does not account sufficiently for gateway-specific operating behavior.

A supervised classifier would depend on engineer-review labels that are only proxy evidence and are not the official hidden challenge ground truth. Using them as training labels could cause the model to learn the proxy rather than the actual underlying problem.

4. Temporal Training and Prediction Separation
Choice
For each prediction week, the final model uses:

uses a 28-day lookback window;

reserves the most recent 7 days for scoring;

uses the preceding 21 days as the training period.

Gateway statistics are also calculated only from the historical training period.

Why I chose it
This creates a clean temporal separation between training information and the observations being scored.

The model therefore does not use the recent observations to define their own historical baseline.

This better represents the real challenge situation where the system receives historical information and must produce rankings for a future/recent period.

Alternative considered
Calculate statistics and fit the model using the entire available period, including the period being scored.

Why I rejected it
That approach could introduce temporal leakage and make validation unrealistically optimistic. It would also be less representative of a real deployment where future observations are not available during training.

5. Validation Strategy
Choice
I used three additional validation tests:

Unseen gateways

Future weeks

Network-change stress test

Why I chose them
The challenge specifically requires the model to be tested on gateways it has not seen, tested on weeks after training, and evaluated under network changes.

These tests are more realistic than simply checking performance on the same data used for training.

Alternative considered
Use only the February engineer-review comparison.

Why I rejected it
The engineer review is only proxy evidence and does not represent the official hidden challenge ground truth.

A model could perform well on that reviewed sample without generalizing to unseen gateways or future weeks.

Therefore, I added separate tests for unseen gateways, future time periods, and network changes.

Validation Evidence
Baseline vs Final ML
The available engineer-review data was used only as proxy evidence to compare the supplied baseline with the final ML model.

Model	Reviewed predictions	Known Schlecht	Known Normal	Proxy precision	Proxy wasted visits	Proxy visit cost
3-Sigma Baseline	48	22	26	45.83%	26	€9880
Final ML V4	89	85	4	95.51%	4	€1520
Proxy cost improvement
Baseline proxy cost : €9880
Final ML proxy cost : €1520
Cost reduction      : €8360
Percentage reduction: 84.62%
Important: These are proxy results only. They are not the official hidden challenge score.

Unseen Gateway Test
The model was tested on 64 gateways that were not included in the training gateway set.

Training gateways    : 256
Unseen test gateways : 64

Reviewed selections : 59
Known Schlecht      : 46
Known Normal        : 13
Proxy precision     : 77.97%
Proxy wasted visits : 13
Proxy visit cost    : €4940
This provides evidence that the model can produce useful rankings for gateways that were not present in its training gateway set.

These results are based on proxy engineer-review information.

Future-Week Test
The model was tested by training on an earlier historical period and scoring a later week.

Reviewed selections : 93
Known Schlecht      : 91
Known Normal        : 2
Proxy precision     : 97.85%
Proxy wasted visits : 2
Proxy visit cost    : €760
This tests temporal generalization rather than simply testing on observations from the training period.

These results are proxy evidence only.

Network-Change Stress Test
A controlled stress test was used to examine the model's behavior when network conditions change.

Normal average anomaly score : 0.364645
Changed average anomaly score: 0.374168

Top-15 overlap: 13 / 15

Gateways affected                 : 61
Changed gateways in top 15        : 5
Percentage of top 15 affected     : 33.33%
Average rank improvement affected : 11.33
The result shows that changed gateways receive somewhat higher anomaly scores while the ranking retains substantial overlap with the original top 15.

This is a controlled stress test and is not official challenge scoring.

Limitations — What It Cannot Do
The solution has several limitations.

1. No official hidden labels
The model is not trained against the official hidden challenge ground truth.

Engineer-review information was used only for proxy evaluation.

2. Limited feature set
The final model uses only three telemetry metrics. Other available telemetry signals may contain additional information that could improve the ranking.

3. Anomaly detection is not diagnosis
A high Isolation Forest anomaly score means that a gateway's behavior is unusual compared with its historical pattern. It does not identify the exact physical or network cause of the problem.

4. Sparse or new gateways
Gateways with little historical data may have less reliable gateway-relative statistics.

5. Network-change test is controlled
The network-change experiment is a controlled stress test and cannot represent every possible real-world network change.

6. No online retraining during prediction
The solution intentionally separates training from prediction. It does not retrain itself while answering a live prediction request because the challenge requires prediction to finish quickly.

What Another Two Weeks Would Fix
With another two weeks, I would focus on:

Testing additional telemetry features while avoiding leakage.

Comparing several anomaly-detection models.

Improving handling of gateways with sparse historical data.

Performing stronger time-based model selection.

Testing more realistic network-change scenarios.

Improving explanations for why individual gateways receive high anomaly scores.

Investigating whether gateway groups have different normal operating patterns.

The goal would be to improve robustness and generalization without making the solution unnecessarily complex.

Summary
The final design prioritizes:

simple and explainable inputs;

gateway-relative behavior;

temporal separation between training and prediction;

unsupervised anomaly detection;

testing on unseen gateways;

testing on future weeks;

testing under network changes;

and cost-oriented evaluation.

The solution is intentionally kept relatively simple so that it can be explained, reproduced, and run during the live challenge.

