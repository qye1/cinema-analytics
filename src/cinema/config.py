# общие константы проекта: календарь, цены, список каналов и все, что заложено в симуляцию

from datetime import date, timedelta

START = date(2024, 1, 1)                # понедельник
NW = 104                                # недель в данных
END = START + timedelta(weeks=NW)       # граница данных, не включительно
DAYS = NW * 7

HIKE_WEEK = 61                          # 3 марта 2025, повышение цены в части регионов
HIKE_DATE = START + timedelta(weeks=HIKE_WEEK)
HIKE_FACTOR = 1.15
HIKE_LOG_HAZARD = 0.28                  # во сколько раз вырастает риск ухода после повышения цены, в логарифмах
EXP_WEEK = 71                           # 12 мая 2025, эксперимент с retention оффером
EXP_DATE = START + timedelta(weeks=EXP_WEEK)
EXP_NAME = "retention_offer_2025_05"
INCIDENT_DAYS = [date(2025, 3, 10), date(2025, 3, 11), date(2025, 3, 12)]   # сбой платежного шлюза
INCIDENT_FAIL_P = 0.45

PLANS = ["basic", "standard", "premium"]
PRICE = {"basic": 299, "standard": 499, "premium": 799}       # цена за цикл в 4 недели, условный месяц
ANNUAL_MONTHS = 10                       # годовая подписка стоит как 10 циклов
CYCLE_WEEKS = 4
TRIAL_WEEKS = 2

CHANNELS = ["organic", "search", "social", "referral", "partner"]
REGIONS = ["Moscow", "SPb", "Central", "Volga", "South", "Ural", "Siberia", "FarEast"]
REGION_P = [0.22, 0.10, 0.14, 0.16, 0.12, 0.08, 0.11, 0.07]
TREATED_REGIONS = ["Volga", "South", "Siberia"]
DEVICES = ["tv", "mobile", "web"]
DEVICE_P = [0.40, 0.35, 0.25]

PERSONAS = ["binge", "casual", "weekend", "light"]
PERSONA_P = [0.20, 0.35, 0.25, 0.20]
PERSONA_HOURS = [1.6, 0.9, 1.0, 0.45]
PERSONA_HPS = [2.0, 0.6, 1.1, 0.8]       # часов на сессию
PERSONA_TPS = [0.5, 1.2, 0.9, 1.0]       # названий на сессию
PERSONA_WEEKEND = [0.45, 0.30, 0.70, 0.30]

# латентные типы реакции на оффер: persuadable, sure_thing, lost_cause, sleeping_dog
USER_TYPES = ["persuadable", "sure_thing", "lost_cause", "sleeping_dog"]
COMPLETE_BASE = [0.80, 0.05, 0.97, 0.50]     # вероятность, что спад интереса закончится уходом
COMPLETE_OFFER = [0.20, 0.05, 0.97, 0.85]    # то же, если человеку показали оффер

# цена привлечения одного регистрации по каналам, руб
SPEND_PER_SIGNUP = {"organic": 0.0, "search": 250.0, "social": 330.0, "referral": 120.0, "partner": 170.0}

PHASE_FACTOR = [1.0, 0.75, 0.5, 0.3]
CHURN_HORIZON_WEEKS = 4

# дни рекламных кампаний, их маркетинг знает заранее, поэтому прогноз может использовать их как признак
CAMPAIGN_DAYS = [START + timedelta(days=i) for i in list(range(333, 337)) + list(range(362, 372))]
