# строит чистые представления и витрины в базе и отдает результаты sql запросов как таблицы pandas

from . import db
from .db import SQL_DIR


def build(conn):
    # сначала представления, потом снимок по месяцам, который зависит от них
    db.run_script(conn, SQL_DIR / "02_views.sql")
    db.run_script(conn, SQL_DIR / "03_user_month.sql")


def quality(conn):
    return db.query_file(conn, SQL_DIR / "01_quality.sql")


def mrr_waterfall(conn):
    return db.query_file(conn, SQL_DIR / "04_mrr_waterfall.sql")


def cohorts(conn):
    return db.query_file(conn, SQL_DIR / "05_cohorts.sql")


def funnel_cac(conn):
    return db.query_file(conn, SQL_DIR / "06_funnel_cac.sql")


def payment_failures(conn):
    return db.query_file(conn, SQL_DIR / "07_payment_failures.sql")


def daily_signups(conn):
    return db.query_file(conn, SQL_DIR / "08_daily_signups.sql")


def region_weekly_churn(conn):
    return db.query_file(conn, SQL_DIR / "09_region_weekly_churn.sql")


def trial_engagement(conn):
    return db.query_file(conn, SQL_DIR / "10_trial_engagement.sql")


def user_features(conn):
    return db.query_file(conn, SQL_DIR / "11_user_features.sql")
