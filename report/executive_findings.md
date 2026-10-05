# Executive findings

- Analyzed **8,279** published CFPB complaints from Arizona across checking/savings, credit card, prepaid card, and money-transfer/digital-wallet products for calendar years 2024-2025.
- Published complaint volume **increased 46.7%** from 3,356 in 2024 to 4,923 in 2025.
- **Credit card** was the largest product group with 3,458 records.
- The most common issue was **Managing an account**, representing 1,655 complaints (20.0% of the analyzed records).
- Companies marked **99.34%** of complaints as timely responses; 24.01% were closed with monetary or non-monetary relief.
- Consumer narratives were available for 59.29% of records, limiting text-based conclusions to the opt-in subset.
- A January 2025 money-transfer/digital-wallet spike added **726** complaints versus January 2024 and represented **46.3%** of the total year-over-year increase. Two companies accounted for **92.0%** of that month's product complaints.
- The spike contained **257 repeated narrative texts** attached to unique complaint IDs. The project preserved those records and flagged the repetition for review instead of assuming they were duplicate complaints.

## Operational recommendations

1. Prioritize root-cause review for the top three issues within each product instead of treating all complaints as one queue.
2. Investigate concentrated spikes and repeated narratives before interpreting a volume change as a broad market trend.
3. Track complaint volume, response category, and timeliness by product monthly; use volume as a workload indicator, not as a standalone quality ranking.
4. Add denominator data such as accounts, transactions, or market share before comparing companies or geographies.

## Interpretation limits

The CFPB states that the complaint database is not a statistical sample of all consumers' experiences. Complaint counts are influenced by company size, product usage, consumer reporting behavior, and publication rules. This project describes the published records and does not rank company quality.

## Forecasting and relief-likelihood modeling

- **Weekly workload forecast:** Across 26 rolling weekly origins, the 8-week median achieved 10.17 complaints/week MAE, 14.2% MAPE, and 12.7% WAPE. This was 22.9% lower MAE than the last-week benchmark. The lagged Ridge model did not beat the median.
- **Relief-likelihood triage:** On the held-out Jul-Dec 2025 period, the selected XGBoost classifier achieved F1 0.46, recall 0.47, and PR-AUC 0.46, versus F1 0.44 for logistic regression. The top-scored 20% contained 42% of complaints recorded as ending in relief.
- Model selection and threshold tuning used Apr-Jun 2025 validation data. Jul-Dec 2025 test data was held out until evaluation. Results are specific to this fixed CFPB snapshot and pinned software versions.
- These are research results. Company-reported relief is not an independent measure of consumer harm. ZIP3 and Older American / Servicemember tags may create fairness concerns; any operational design should assess them and remove or restrict them before deployment.
