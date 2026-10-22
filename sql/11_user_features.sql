/* поведение человека в первые восемь недель после регистрации, из этого строится сегментация
   берем только тех, у кого видно хотя бы четыре недели */

WITH ranked AS (
    SELECT e.*,
           ROW_NUMBER() OVER (PARTITION BY e.user_id ORDER BY e.week_start) AS rn
    FROM engagement_clean e
)
SELECT user_id,
       COUNT(*) AS weeks_seen,
       AVG(hours) AS mean_hours,
       AVG(hours * hours) - AVG(hours) * AVG(hours) AS var_hours,
       SUM(hours) AS sum_hours,
       SUM(sessions) AS sum_sessions,
       SUM(titles) AS sum_titles,
       SUM(hours * weekend_share) / NULLIF(SUM(hours), 0) AS weekend_share
FROM ranked
WHERE rn <= 8
GROUP BY user_id
HAVING COUNT(*) >= 4;
