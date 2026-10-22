/* недельная картина оттока по регионам: сколько платящих на начало недели и сколько из них ушло
   берем только помесячные подписки, у годовых уход возможен только раз в год */

WITH RECURSIVE weeks(week_idx, week_start) AS (
    SELECT 0, '2024-01-01'
    UNION ALL
    SELECT week_idx + 1, date(week_start, '+7 day') FROM weeks WHERE week_idx < 103
),
spells AS (
    SELECT s.user_id, s.start_date, s.end_date_clean AS end_date, s.end_reason, u.region
    FROM subscriptions_clean s
    JOIN users_clean u ON u.user_id = s.user_id
    WHERE s.billing = 'monthly'
)
SELECT w.week_idx, w.week_start, sp.region,
       COUNT(*) AS active_at_start,
       SUM(CASE WHEN sp.end_reason = 'voluntary'
                 AND sp.end_date >= w.week_start AND sp.end_date < date(w.week_start, '+7 day')
                THEN 1 ELSE 0 END) AS churned
FROM weeks w
JOIN spells sp
  ON sp.start_date <= w.week_start
 AND (sp.end_date IS NULL OR sp.end_date >= w.week_start)
GROUP BY w.week_idx, w.week_start, sp.region
ORDER BY w.week_idx, sp.region;
