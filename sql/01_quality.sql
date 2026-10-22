/* проверки качества данных
   каждая строка результата: название проверки, сколько нашлось проблемных строк и что это значит */

SELECT 'duplicate_payments' AS check_name,
       COUNT(*) - COUNT(DISTINCT user_id || '|' || pay_date || '|' || amount || '|' || status || '|' || attempt) AS problem_rows,
       'один и тот же платеж записан дважды под разными номерами' AS meaning
FROM payments

UNION ALL
SELECT 'zero_amount_success', COUNT(*),
       'успешный платеж с суммой ноль или меньше'
FROM payments
WHERE status = 'success' AND amount <= 0

UNION ALL
SELECT 'payment_before_signup', COUNT(*),
       'платеж датирован раньше регистрации человека'
FROM payments p
JOIN users u ON u.user_id = p.user_id
WHERE p.pay_date < u.signup_date

UNION ALL
SELECT 'null_region', COUNT(*),
       'у пользователя не заполнен регион'
FROM users
WHERE region IS NULL

UNION ALL
SELECT 'negative_hours', COUNT(*),
       'отрицательное время просмотра за неделю'
FROM engagement
WHERE hours < 0

UNION ALL
SELECT 'overlapping_spells', COUNT(*),
       'подписка закончилась позже, чем началась следующая у того же человека'
FROM (
    SELECT end_date,
           LEAD(start_date) OVER (PARTITION BY user_id ORDER BY start_date, spell_id) AS next_start
    FROM subscriptions
)
WHERE end_date IS NOT NULL AND next_start IS NOT NULL AND next_start < end_date

UNION ALL
SELECT 'success_payment_outside_spell', COUNT(*),
       'успешный платеж не попадает ни в один период подписки'
FROM payments p
WHERE p.status = 'success'
  AND NOT EXISTS (
      SELECT 1 FROM subscriptions s
      WHERE s.user_id = p.user_id
        AND s.start_date <= p.pay_date
        AND (s.end_date IS NULL OR s.end_date > p.pay_date)
  );
