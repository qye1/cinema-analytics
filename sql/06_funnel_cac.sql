/* воронка и окупаемость каналов
   регистрация, активация за три дня, оплата после триала, затраты, цена платящего клиента,
   выручка за первые 180 дней и отношение этой выручки к цене привлечения */

WITH first_paid AS (
    SELECT user_id, MIN(start_date) AS first_paid_date
    FROM subscriptions_clean
    GROUP BY user_id
),
rev180 AS (
    SELECT u.user_id, COALESCE(SUM(p.amount), 0) AS revenue_180
    FROM users_clean u
    LEFT JOIN payments_clean p
           ON p.user_id = u.user_id AND p.status = 'success'
          AND p.pay_date < date(u.signup_date, '+180 day')
    GROUP BY u.user_id
),
per_user AS (
    SELECT u.channel, u.user_id, u.activated_3d,
           CASE WHEN f.user_id IS NOT NULL THEN 1 ELSE 0 END AS paid,
           r.revenue_180
    FROM users_clean u
    LEFT JOIN first_paid f ON f.user_id = u.user_id
    JOIN rev180 r ON r.user_id = u.user_id
    WHERE u.signup_date < date('2025-06-30')          /* у всех по 180 дней наблюдения */
),
spend AS (
    SELECT channel, SUM(spend) AS spend
    FROM marketing_spend
    WHERE date < date('2025-06-30')
    GROUP BY channel
)
SELECT p.channel,
       COUNT(*) AS signups,
       SUM(p.activated_3d) AS activated,
       SUM(p.paid) AS paying,
       ROUND(100.0 * SUM(p.activated_3d) / COUNT(*), 1) AS activation_pct,
       ROUND(100.0 * SUM(p.paid) / COUNT(*), 1) AS conversion_pct,
       ROUND(COALESCE(s.spend, 0), 0) AS spend,
       ROUND(COALESCE(s.spend, 0) / NULLIF(SUM(p.paid), 0), 0) AS cac_per_paying,
       ROUND(SUM(p.revenue_180) / NULLIF(SUM(p.paid), 0), 0) AS revenue_180_per_paying,
       ROUND(SUM(p.revenue_180) / NULLIF(COALESCE(s.spend, 0), 0), 2) AS revenue180_to_cac
FROM per_user p
LEFT JOIN spend s ON s.channel = p.channel
GROUP BY p.channel
ORDER BY revenue180_to_cac DESC;
