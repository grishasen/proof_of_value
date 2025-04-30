from itertools import product

RECURRING_COST = "RecurringCost"
RECURRING_PERIOD = "RecurringPeriod"
NAME = "Name"
RANK = "Rank"
CUSTOMER_ID = "SubjectID"
OUTCOME = "Outcome"
MODELCONTROLGROUP = "ModelControlGroup"
REVENUE_PROP_NAME = "Revenue"
PURCHASED_DATE = "PurchasedDate"
CLV_MODEL = "non-contractual"
ONE_TIME_COST = "OneTimeCost"
HOLDING_ID = "HoldingID"
INTERACTION_ID = "InteractionID"

_scores = [''.join(p) for p in product('1234', repeat=3)]

_default_rfm_config_retail = {}
_champ = [s for s in _scores if int(s[0]) >= 3 and int(s[1]) >= 3 and int(s[2]) >= 3]
_loyal = [s for s in _scores
          if int(s[0]) >= 2 and int(s[1]) >= 3 and int(s[2]) >= 2
          and s not in _champ]
_pot = [s for s in _scores
        if int(s[0]) >= 3 and int(s[1]) >= 2 and int(s[2]) >= 2
        and s not in _champ]
_risk = [s for s in _scores if int(s[0]) <= 2 and int(s[1]) >= 2 and int(s[2]) >= 2]
_dorm = [s for s in _scores if s not in (_champ + _loyal + _pot + _risk)]

_default_rfm_config_retail.update({
    "Wealth Champions": _champ,
    "Premier Clients": _loyal,
    "Growth Investors": _pot,
    "At-Risk Clients": _risk,
    "Dormant Accounts": _dorm,
})

_default_rfm_config_telco = {}
_champ = [s for s in _scores if int(s[0]) >= 3 and int(s[1]) >= 3 and int(s[2]) >= 3]
_loyal = [s for s in _scores
          if int(s[0]) >= 2 and int(s[1]) >= 3 and int(s[2]) >= 2
          and s not in _champ]
_pot = [s for s in _scores
        if int(s[0]) >= 3 and int(s[1]) >= 2 and int(s[2]) >= 2
        and s not in _champ]
_risk = [s for s in _scores if int(s[0]) <= 2 and int(s[1]) >= 2 and int(s[2]) >= 2]
_dorm = [s for s in _scores if s not in (_champ + _loyal + _pot + _risk)]

_default_rfm_config_telco.update({
    "Platinum Subscribers": _champ,
    "Engaged Subscribers": _loyal,
    "Potential Upsell Group": _pot,
    "Churn Risk": _risk,
    "Dormant Lines": _dorm,
})

_default_rfm_config_ecomm = {}
_champ = [s for s in _scores if int(s[0]) >= 3 and int(s[1]) >= 3 and int(s[2]) >= 3]
_loyal = [s for s in _scores
          if int(s[0]) >= 2 and int(s[1]) >= 3 and int(s[2]) >= 2
          and s not in _champ]
_pot = [s for s in _scores
        if int(s[0]) >= 3 and int(s[1]) >= 2 and int(s[2]) >= 2
        and s not in _champ]
_risk = [s for s in _scores if int(s[0]) <= 2 and int(s[1]) >= 2 and int(s[2]) >= 2]
_dorm = [s for s in _scores if s not in (_champ + _loyal + _pot + _risk)]

_default_rfm_config_ecomm.update({
    "Champions": _champ,
    "Repeat Buyers": _loyal,
    "High-Potential Buyers": _pot,
    "At-Risk Shoppers": _risk,
    "Inactive Shoppers": _dorm,
})

rfm_config_dict = {'retail_banking': _default_rfm_config_retail,
                   'telco': _default_rfm_config_telco,
                   'e-commerce': _default_rfm_config_ecomm}
