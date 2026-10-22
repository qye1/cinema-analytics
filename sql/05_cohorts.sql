/* когорты по месяцу первой оплаты: сколько человек и сколько денег остается спустя n месяцев */

WITH first_month AS (
    SELECT user_id, MIN(month_idx) AS cohort_idx FROM user_month GROUP BY user_id
),
cohort_size AS (
    SELECT cohort_idx, COUNT(*) AS cohort_users FROM first_month GROUP BY cohort_idx
),
cohort_start_mrr AS (
    SELECT f.cohort_idx, SUM(u.mrr) AS start_mrr
    FROM first_month f
    JOIN user_month u ON u.user_id = f.user_id AND u.month_idx = f.cohort_idx
    GROUP BY f.cohort_idx
),
activity AS (
    SELECT f.cohort_idx, u.month_idx - f.cohort_idx AS age,
           COUNT(DISTINCT u.user_id) AS active_users, SUM(u.mrr) AS mrr
    FROM first_month f
    JOIN user_month u ON u.user_id = f.user_id
    GROUP BY f.cohort_idx, u.month_idx - f.cohort_idx
)
SELECT a.cohort_idx, a.age, s.cohort_users, a.active_users, a.mrr,
       ROUND(100.0 * a.active_users / s.cohort_users, 2) AS retention_pct,
       ROUND(100.0 * a.mrr / c.start_mrr, 2) AS revenue_retention_pct
FROM activity a
JOIN cohort_size s ON s.cohort_idx = a.cohort_idx
JOIN cohort_start_mrr c ON c.cohort_idx = a.cohort_idx
ORDER BY a.cohort_idx, a.age;
