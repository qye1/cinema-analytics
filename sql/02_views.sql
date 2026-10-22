/* чистые представления: дальше вся аналитика читает только их */

DROP VIEW IF EXISTS users_clean;
CREATE VIEW users_clean AS
SELECT user_id, signup_date, channel,
       COALESCE(region, 'unknown') AS region,
       device, plan_at_signup, billing, activated_3d
FROM users;

/* если подписка заходит на следующую, обрезаем ее по началу следующей */
DROP VIEW IF EXISTS subscriptions_clean;
CREATE VIEW subscriptions_clean AS
SELECT spell_id, user_id, plan, billing, start_date,
       CASE
           WHEN end_date IS NULL THEN NULL
           WHEN next_start IS NOT NULL AND next_start < end_date THEN next_start
           ELSE end_date
       END AS end_date_clean,
       end_reason
FROM (
    SELECT s.*,
           LEAD(start_date) OVER (PARTITION BY user_id ORDER BY start_date, spell_id) AS next_start
    FROM subscriptions s
);

/* платежи без дублей, без нулевых успешных и без датированных раньше регистрации */
DROP VIEW IF EXISTS payments_clean;
CREATE VIEW payments_clean AS
WITH ranked AS (
    SELECT p.*,
           ROW_NUMBER() OVER (PARTITION BY user_id, pay_date, amount, status, attempt ORDER BY payment_id) AS rn
    FROM payments p
)
SELECT r.payment_id, r.user_id, r.pay_date, r.amount, r.status, r.attempt, r.plan
FROM ranked r
JOIN users u ON u.user_id = r.user_id
WHERE r.rn = 1
  AND r.pay_date >= u.signup_date
  AND NOT (r.status = 'success' AND r.amount <= 0);

DROP VIEW IF EXISTS engagement_clean;
CREATE VIEW engagement_clean AS
SELECT user_id, week_start, hours, sessions, titles, weekend_share
FROM engagement
WHERE hours >= 0;
