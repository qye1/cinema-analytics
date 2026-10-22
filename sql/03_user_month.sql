/* снимок на конец каждого месяца: у кого активна платная подписка и сколько это в месяц (MRR)
   годовая подписка пересчитывается в месячный эквивалент, цена берется из прайс листа на эту дату */

DROP TABLE IF EXISTS user_month;
CREATE TABLE user_month AS
WITH RECURSIVE months(month_idx, month_end) AS (
    SELECT 0, '2024-01-31'
    UNION ALL
    SELECT month_idx + 1,
           date(month_end, '+1 day', 'start of month', '+1 month', '-1 day')
    FROM months
    WHERE month_idx < 23
),
active AS (
    SELECT s.user_id, m.month_idx, m.month_end, s.plan, s.billing, u.region
    FROM months m
    JOIN subscriptions_clean s
      ON s.start_date <= m.month_end
     AND (s.end_date_clean IS NULL OR s.end_date_clean > m.month_end)
    JOIN users_clean u ON u.user_id = s.user_id
),
priced AS (
    SELECT a.*,
           COALESCE(
               (SELECT p.price FROM price_list p
                WHERE p.region = a.region AND p.plan = a.plan AND p.effective_from <= a.month_end
                ORDER BY p.effective_from DESC LIMIT 1),
               (SELECT p.price FROM price_list p
                WHERE p.region = 'Moscow' AND p.plan = a.plan AND p.effective_from <= a.month_end
                ORDER BY p.effective_from DESC LIMIT 1)
           ) AS price
    FROM active a
)
SELECT user_id, month_idx, month_end, plan, billing, region,
       CASE billing WHEN 'annual' THEN price * 10.0 / 12 ELSE price END AS mrr
FROM priced;

CREATE INDEX ix_um ON user_month (user_id, month_idx);
CREATE INDEX ix_um_m ON user_month (month_idx);
