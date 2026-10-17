"""Appetite constants for the synthetic P&C program (mirrors guidelines.md)."""

TARGET_CLASSES = {
    "332312": ("Fabricated structural metal manufacturing", "medium"),
    "423710": ("Hardware merchant wholesalers", "low"),
    "444110": ("Home centers", "low"),
    "541330": ("Engineering services", "low"),
    "722511": ("Full-service restaurants", "medium"),
    "238220": ("Plumbing and HVAC contractors", "medium"),
    "531120": ("Lessors of nonresidential buildings", "low"),
    "311811": ("Retail bakeries", "medium"),
}
REFERRAL_CLASSES = {
    "325998": ("Miscellaneous chemical manufacturing", "high"),
    "562111": ("Solid waste collection", "high"),
    "336411": ("Aircraft manufacturing", "high"),
}
PROHIBITED_CLASSES = {
    "325920": ("Explosives manufacturing", "high"),
    "211120": ("Crude petroleum extraction", "high"),
}
UNLISTED_EXAMPLES = {
    "713940": ("Fitness and recreational sports centers", "medium"),
    "484121": ("General freight trucking, long-distance", "high"),
}
LICENSED_STATES = {"TX", "OK", "LA", "AR", "NM", "KS", "MO", "CO"}

TIV_REFER = 15_000_000
TIV_DECLINE = 25_000_000
LR_REFER = 0.65
LR_DECLINE = 1.00
LARGE_LOSS = 250_000
MIN_LOSS_YEARS = 3
MIN_YEARS_IN_BUSINESS = 3
OLD_BUILDING_YEAR = 1960
MODEL_REFER_PROB = 0.25
AS_OF_YEAR = 2026


def hazard_of(naics: str | None) -> str:
    for table in (TARGET_CLASSES, REFERRAL_CLASSES, PROHIBITED_CLASSES, UNLISTED_EXAMPLES):
        if naics in table:
            return table[naics][1]
    return "medium"
