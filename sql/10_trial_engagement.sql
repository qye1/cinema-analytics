/* средняя вовлеченность в первые две недели после регистрации, то есть в триале
   это спокойный замер привычного уровня просмотра, до того как человек начнет остывать */

SELECT user_id,
       AVG(hours) AS trial_hours,
       COUNT(*) AS trial_weeks_seen
FROM (
    SELECT e.user_id, e.hours,
           ROW_NUMBER() OVER (PARTITION BY e.user_id ORDER BY e.week_start) AS rn
    FROM engagement_clean e
)
WHERE rn <= 2
GROUP BY user_id;
