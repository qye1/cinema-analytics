/* схема базы подписочного кинотеатра
   даты хранятся текстом в формате ГГГГ-ММ-ДД, так их удобно сравнивать и в sqlite, и в pandas */

CREATE TABLE users (
    user_id        INTEGER PRIMARY KEY,
    signup_date    TEXT NOT NULL,
    channel        TEXT NOT NULL,
    region         TEXT,
    device         TEXT NOT NULL,
    plan_at_signup TEXT NOT NULL,
    billing        TEXT NOT NULL,
    activated_3d   INTEGER NOT NULL
);

/* одна строка это один непрерывный период на одном тарифе, смена тарифа или возвращение дают новую строку */
CREATE TABLE subscriptions (
    spell_id    INTEGER PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    plan        TEXT NOT NULL,
    billing     TEXT NOT NULL,
    start_date  TEXT NOT NULL,
    end_date    TEXT,
    end_reason  TEXT NOT NULL
);

CREATE TABLE payments (
    payment_id INTEGER PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    pay_date   TEXT NOT NULL,
    amount     REAL NOT NULL,
    status     TEXT NOT NULL,
    attempt    INTEGER NOT NULL,
    plan       TEXT NOT NULL
);

CREATE TABLE engagement (
    user_id       INTEGER NOT NULL,
    week_start    TEXT NOT NULL,
    hours         REAL NOT NULL,
    sessions      INTEGER NOT NULL,
    titles        INTEGER NOT NULL,
    weekend_share REAL NOT NULL
);

CREATE TABLE tickets (
    ticket_id    INTEGER PRIMARY KEY,
    user_id      INTEGER NOT NULL,
    created_date TEXT NOT NULL,
    category     TEXT NOT NULL
);

CREATE TABLE marketing_spend (
    date    TEXT NOT NULL,
    channel TEXT NOT NULL,
    spend   REAL NOT NULL
);

CREATE TABLE experiment_assignment (
    user_id       INTEGER NOT NULL,
    experiment    TEXT NOT NULL,
    variant       TEXT NOT NULL,
    assigned_date TEXT NOT NULL,
    hours_ratio   REAL
);

CREATE TABLE price_list (
    region         TEXT NOT NULL,
    plan           TEXT NOT NULL,
    effective_from TEXT NOT NULL,
    price          REAL NOT NULL
);

CREATE INDEX ix_sub_user ON subscriptions (user_id, start_date);
CREATE INDEX ix_pay_user ON payments (user_id, pay_date);
CREATE INDEX ix_pay_date ON payments (pay_date);
CREATE INDEX ix_eng_user ON engagement (user_id, week_start);
CREATE INDEX ix_tick_user ON tickets (user_id, created_date);
