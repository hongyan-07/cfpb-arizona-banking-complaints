-- CFPB Arizona Consumer Banking Complaint Analytics, 2024-2025
-- The processed table is created by src/analyze.py.

-- 1. Validate scope and uniqueness.
SELECT
    COUNT(*) AS rows_in_scope,
    COUNT(DISTINCT complaint_id) AS unique_complaint_ids,
    MIN(date_received) AS first_date,
    MAX(date_received) AS last_date,
    SUM(CASE WHEN issue IS NULL THEN 1 ELSE 0 END) AS missing_issue_rows
FROM complaints;

-- 2. Measure year-over-year volume by product.
SELECT year, product_group, COUNT(*) AS complaints
FROM complaints
GROUP BY year, product_group
ORDER BY year, complaints DESC;

-- 3. Identify the top three issues within each product using a window function.
WITH issue_counts AS (
    SELECT product_group, issue, COUNT(*) AS complaints
    FROM complaints
    WHERE issue IS NOT NULL
    GROUP BY product_group, issue
),
ranked AS (
    SELECT
        product_group,
        issue,
        complaints,
        ROW_NUMBER() OVER (
            PARTITION BY product_group ORDER BY complaints DESC, issue
        ) AS issue_rank
    FROM issue_counts
)
SELECT product_group, issue, complaints, issue_rank
FROM ranked
WHERE issue_rank <= 3
ORDER BY product_group, issue_rank;

-- 4. Calculate response indicators by product.
SELECT
    product_group,
    COUNT(*) AS complaints,
    ROUND(100.0 * AVG(timely_response), 2) AS timely_response_pct,
    ROUND(100.0 * AVG(relief_response), 2) AS relief_response_pct,
    ROUND(100.0 * AVG(has_narrative), 2) AS narrative_available_pct
FROM complaints
GROUP BY product_group
ORDER BY complaints DESC;

-- 5. Create a monthly monitoring table.
SELECT month, product_group, COUNT(*) AS complaints
FROM complaints
GROUP BY month, product_group
ORDER BY month, product_group;
