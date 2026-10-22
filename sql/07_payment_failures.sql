/* доля неудачных первых попыток оплаты по дням, из нее потом ищем сбои шлюза */

SELECT pay_date AS day,
       COUNT(*) AS attempts,
       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
       ROUND(1.0 * SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) / COUNT(*), 4) AS fail_rate
FROM payments_clean
WHERE attempt = 1
GROUP BY pay_date
ORDER BY pay_date;
