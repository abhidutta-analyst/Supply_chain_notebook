# Supply Chain Delivery Performance Analysis

End-to-end analysis of an e-commerce supply chain dataset to understand **why deliveries run late**, **what delays cost the business**, and whether late deliveries can be **predicted before they happen**.

## Business Problem

A global e-commerce company selling sporting goods manages order fulfilment and shipping across multiple regions. Delivery performance is inconsistent: actual shipping times frequently deviate from the promised schedule, producing late deliveries and unpredictable order profitability.

**Goal:** analyze delivery operations, identify the bottlenecks behind late deliveries, quantify the profit impact, and build a machine learning model that flags orders at risk of arriving late.

## Dataset

| | |
|---|---|
| **File** | `DataCoSupplyChainDataset.csv` |
| **Rows × columns (raw)** | 27,836 × 53 |
| **After cleaning** | 27,058 × 20 |
| **Period covered** | January 2015 – February 2018 |
| **Grain** | One row per order item, with customer, product, shipping and payment attributes |

The raw file contained identifier, PII (customer names, emails, passwords), and redundant columns that were dropped. Orders with `Shipping canceled` status were removed since they never enter the delivery pipeline. Two date fields were converted to datetime, and four analysis features were engineered:

- **Order Processing Time** – days between order placement and shipping date
- **Delay** – processing time minus scheduled shipping days (negative = early)
- **Is_Delayed** – whether the delay is positive
- **Profitability Flag** – Profit / Loss / Break-even per order

## Key Insights

### 1. Late delivery is the norm, not the exception

- **56.6% of orders ship late** (15,329 of 27,058); only 43.4% arrive on time or early
- The **most common outcome is exactly 1 day late — 35.3% of all orders**
- ~24.4% of orders actually ship **early** (1–2 days ahead of schedule)
- 90% of delays are **3 days or fewer** — this is a systematic scheduling problem, not a long-tail logistics failure

### 2. The scheduling system is miscalibrated, especially for First Class

| Shipping Mode | Scheduled days | Actual behaviour | Delay rate |
|---|---|---|---|
| **First Class** | 1 day | Always takes 2 days | **100% late** |
| **Second Class** | ~2 days | Usually takes longer | **79.8% late** |
| **Standard Class** | 4 days | Often beats the clock | **39.4% late** |
| **Same Day** | 0 days | Meets the promise | **0% late** |

Every single First Class shipment is scheduled for 1 day and takes 2. The premium ("fast") tiers consistently under-deliver on their promise, while Standard Class routinely ships early. Simply re-baselining First and Second Class schedules would eliminate most recorded lateness overnight — and would let the company set honest delivery promises at checkout.

### 3. Delays are not (yet) destroying margin — but loss-making orders are widespread

- Mean profit per order is roughly flat across delay levels (~$21–25), dipping only slightly for 3-day-late orders ($17.8)
- Orders that shipped late still generated **~$331.7K in net profit**, so the cost of lateness shows up as a service/reliability problem rather than an immediate margin hit
- The bigger profitability issue: **18.6% of all orders (5,046) lose money outright**, i.e. sold below cost

### 4. Payment friction, not customer mix, drives lateness

- Orders in **PAYMENT_REVIEW (76.2%)** and **PENDING_PAYMENT (67.1%)** status are the most delayed — stuck payment flows cascade into late shipments
- **PAYMENT transaction type: 67.5% late** vs **TRANSFER: 47.2%**
- Customer segment is nearly irrelevant (all three segments cluster at 56–57%) — so targeting interventions at customers is the wrong lever

### 5. Regional and seasonal patterns are real but narrow

- Worst regions: **East Africa (62.0%)**, **Central Asia (60.4%)**, **South of USA (60.0%)**; best among the top-10 list: Western Europe (57.3%)
- Worst month: **August (60.3%)**; best: January (53.7%)
- Day of week and hour of day show little variation — lateness is driven by *how* orders ship, not *when* they are placed
- A drill-down into Central Africa confirmed the global pattern: First Class shipping (100% late), pending-payment status (76.9%) and PAYMENT type (75.0%) top the driver list

## Machine Learning: Late Delivery Prediction

A Random Forest classifier predicts the dataset's `Late_delivery_risk` flag (1 = order will ship late) using only information available **at or before dispatch** — no leakage from actual shipping times.

**Features:** payment type, scheduled shipping days, product category, customer segment, department, order region, shipping mode, order month, order hour.

**Pipeline:**
1. Categorical features frequency-encoded (robust for high-cardinality columns)
2. 80/20 stratified train/test split
3. **SMOTE** oversampling on the training set only (12,686 vs 8,959 → balanced)
4. `RandomForestClassifier` (scikit-learn)

**Results on the held-out test set:**

| Metric | Score |
|---|---|
| Accuracy | **0.74** |
| Precision (late class) | **0.79** |
| Recall (late class) | **0.75** |
| F1 (late class) | **0.77** |
| Baseline (majority class) | 0.59 |

The model lifts accuracy ~15 points over naively guessing "late" every time, with precision weighted toward the business-critical class: when it flags an order as at-risk, it is right ~4 times out of 5. With 56.6% of orders genuinely running late, this gives operations a usable early-warning signal for proactive intervention.

## Repository Structure

```
Supply_Chain/
├── README.md                        # This file
├── Supply_Chain_Analysis.ipynb      # Full analysis: EDA → KPIs → ML
└── DataCoSupplyChainDataset.csv     # Source data
```

## How to Run

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Supply_Chain_Analysis.ipynb
   ```
4. Run all cells — the CSV is read from the project root

## Tools & Techniques

**Python** · pandas · NumPy · matplotlib · seaborn · scikit-learn · imbalanced-learn (SMOTE)

EDA · KPI definition · delay & profitability feature engineering · bottleneck analysis by region / mode / status / department · time-based seasonality analysis · class-imbalance handling · Random Forest classification

## Recommendations

1. **Re-baseline First and Second Class delivery promises** — the single highest-impact fix, converting a 100%/80% lateness problem into honest scheduling
2. **Fix the payment review bottleneck** — expedite or parallelise payment verification for PAYMENT-type orders before they enter the shipping queue
3. **Review pricing on the 18.6% of orders sold at a loss** — discounting and shipping cost erosion is a larger profit leak than the delays themselves
4. **Deploy the delay-risk model at order entry** so at-risk orders get proactive customer communication or a faster fulfilment path

## Future Work

- Feature importance / SHAP analysis to make the model's drivers transparent
- Gradient boosting (XGBoost / LightGBM) benchmark and probability calibration for risk scoring
- Separate "will be late" from "how many days late" (regression head) to prioritise the worst cases
- Profit-aware model: predict expected profit impact, not just lateness
