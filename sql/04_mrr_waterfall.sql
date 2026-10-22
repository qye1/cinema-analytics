/* движение MRR по месяцам: откуда пришли деньги и куда ушли
   new: человек платит впервые, reactivation: вернулся после перерыва,
   expansion и contraction: сменил тариф вверх или вниз,
   price_uplift: тариф тот же, а сумма выросла из за повышения цены,
   churn: перестал платить */

WITH months AS (
    SELECT DISTINCT month_idx, month_end FROM user_month
),
first_month AS (
    SELECT user_id, MIN(month_idx) AS first_idx FROM user_month GROUP BY user_id
),
pairs AS (
    SELECT COALESCE(cur.user_id, prev.user_id) AS user_id,
           COALESCE(cur.month_idx, prev.month_idx + 1) AS month_idx,
           COALESCE(prev.mrr, 0) AS prev_mrr,
           COALESCE(cur.mrr, 0) AS cur_mrr,
           prev.plan AS prev_plan,
           cur.plan AS cur_plan
    FROM user_month cur
    LEFT JOIN user_month prev ON prev.user_id = cur.user_id AND prev.month_idx = cur.month_idx - 1
    UNION ALL
    SELECT prev.user_id, prev.month_idx + 1, prev.mrr, 0, prev.plan, NULL
    FROM user_month prev
    LEFT JOIN user_month cur ON cur.user_id = prev.user_id AND cur.month_idx = prev.month_idx + 1
    WHERE cur.user_id IS NULL AND prev.month_idx < 23
),
classified AS (
    SELECT p.month_idx,
           CASE
               WHEN p.prev_mrr = 0 AND p.cur_mrr > 0 AND f.first_idx = p.month_idx THEN 'new'
               WHEN p.prev_mrr = 0 AND p.cur_mrr > 0 THEN 'reactivation'
               WHEN p.prev_mrr > 0 AND p.cur_mrr = 0 THEN 'churn'
               WHEN p.cur_plan = p.prev_plan AND ABS(p.cur_mrr - p.prev_mrr) > 1e-9 THEN 'price_uplift'
               WHEN p.cur_mrr > p.prev_mrr THEN 'expansion'
               WHEN p.cur_mrr < p.prev_mrr THEN 'contraction'
               ELSE 'flat'
           END AS kind,
           p.cur_mrr - p.prev_mrr AS delta
    FROM pairs p
    JOIN first_month f ON f.user_id = p.user_id
)
SELECT m.month_idx, m.month_end,
       COALESCE((SELECT SUM(mrr) FROM user_month u WHERE u.month_idx = m.month_idx - 1), 0) AS start_mrr,
       COALESCE(SUM(CASE WHEN c.kind = 'new' THEN c.delta END), 0) AS new_mrr,
       COALESCE(SUM(CASE WHEN c.kind = 'reactivation' THEN c.delta END), 0) AS reactivation_mrr,
       COALESCE(SUM(CASE WHEN c.kind = 'expansion' THEN c.delta END), 0) AS expansion_mrr,
       COALESCE(SUM(CASE WHEN c.kind = 'price_uplift' THEN c.delta END), 0) AS price_uplift_mrr,
       COALESCE(SUM(CASE WHEN c.kind = 'contraction' THEN c.delta END), 0) AS contraction_mrr,
       COALESCE(SUM(CASE WHEN c.kind = 'churn' THEN c.delta END), 0) AS churn_mrr,
       (SELECT COALESCE(SUM(mrr), 0) FROM user_month u WHERE u.month_idx = m.month_idx) AS end_mrr,
       (SELECT COUNT(*) FROM user_month u WHERE u.month_idx = m.month_idx) AS end_subscribers
FROM months m
LEFT JOIN classified c ON c.month_idx = m.month_idx
GROUP BY m.month_idx, m.month_end
ORDER BY m.month_idx;
