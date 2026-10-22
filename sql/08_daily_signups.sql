/* ежедневные регистрации, из них строится прогноз */

SELECT signup_date AS day, COUNT(*) AS signups,
       SUM(CASE WHEN channel = 'social' THEN 1 ELSE 0 END) AS social_signups
FROM users_clean
GROUP BY signup_date
ORDER BY signup_date;
