"""Generate the SYNTHETIC example records for the vertical starter kits.

All names, amounts and events are invented. Run: python examples/make_examples.py
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).parent

INS_Q = {
    "disposition": {
        "type": "choice",
        "instructions": "Decide what the adjuster should do with this claim next.",
        "criteria": {
            "approve": "Covered loss, documentation complete, amount consistent with evidence; pay the claim.",
            "deny": "Loss is clearly excluded or outside the policy period / coverage.",
            "escalate": "Possible fraud, litigation, injury, or amount above the desk authority limit; send to SIU or a senior adjuster.",
            "request_info": "Likely covered but documentation is missing or inconsistent; ask the claimant for more information.",
        },
    },
    "fraud_indicators": {
        "type": "noul",
        "instructions": "Are there one or more recognised fraud red flags (recent policy inception, prior similar claims, inconsistent dates, inflated invoices)?",
    },
    "severity": {
        "type": "score",
        "instructions": "Rate the financial and human severity of the loss.",
        "criteria": [
            "Minor: under $1,000, no injury",
            "Moderate: $1,000-$10,000, no injury",
            "Serious: $10,000-$50,000 or minor injury",
            "Severe: over $50,000 or significant injury",
            "Catastrophic: total loss, fatality, or multiple serious injuries",
        ],
    },
    "coverage_category": {
        "type": "choice",
        "instructions": "Which coverage section does this claim fall under?",
        "criteria": {
            "auto_collision": "Damage to the insured vehicle from a collision.",
            "auto_liability": "Third-party bodily injury or property damage caused by the insured driver.",
            "property_water": "Water damage to a dwelling (burst pipe, leak, flood if endorsed).",
            "property_fire": "Fire or smoke damage to a dwelling or contents.",
            "theft": "Theft of insured property.",
        },
    },
}

INS_CASES = [
    # (state, labels)
    ({"policy": {"type": "auto", "inception": "2022-03-01", "deductible": 500}, "claim": {"date_of_loss": "2026-08-14", "description": "Rear-ended at a stop light; bumper and tail light damage.", "estimate_usd": 2300, "police_report": True, "photos": 6, "injuries": "none", "prior_claims_3y": 0}}, {"disposition": "approve", "fraud_indicators": False, "severity": 1, "coverage_category": "auto_collision"}),
    ({"policy": {"type": "home", "inception": "2019-06-10", "deductible": 1000}, "claim": {"date_of_loss": "2026-07-02", "description": "Kitchen pipe burst overnight; flooring and lower cabinets damaged.", "estimate_usd": 8700, "plumber_invoice": True, "photos": 12, "prior_claims_3y": 0}}, {"disposition": "approve", "fraud_indicators": False, "severity": 1, "coverage_category": "property_water"}),
    ({"policy": {"type": "home", "inception": "2026-07-28", "deductible": 1000}, "claim": {"date_of_loss": "2026-08-03", "description": "Laptop, two watches and jewellery stolen; no forced entry.", "estimate_usd": 14800, "police_report": False, "receipts": "none", "prior_claims_3y": 2}}, {"disposition": "escalate", "fraud_indicators": True, "severity": 2, "coverage_category": "theft"}),
    ({"policy": {"type": "auto", "inception": "2021-01-15", "deductible": 250, "status": "lapsed 2026-05-01 for non-payment"}, "claim": {"date_of_loss": "2026-06-20", "description": "Side-swiped in parking lot.", "estimate_usd": 1900}}, {"disposition": "deny", "fraud_indicators": False, "severity": 1, "coverage_category": "auto_collision"}),
    ({"policy": {"type": "home", "inception": "2015-09-01", "deductible": 2500}, "claim": {"date_of_loss": "2026-09-11", "description": "Electrical fire in garage spread to the house; family displaced.", "estimate_usd": 310000, "fire_dept_report": True, "injuries": "one smoke inhalation, treated and released"}}, {"disposition": "escalate", "fraud_indicators": False, "severity": 4, "coverage_category": "property_fire"}),
    ({"policy": {"type": "auto", "inception": "2020-11-02", "deductible": 500}, "claim": {"date_of_loss": "2026-08-30", "description": "Insured ran a red light and struck a cyclist; cyclist hospitalised with a fractured leg.", "police_report": True, "injuries": "third party, fracture"}}, {"disposition": "escalate", "fraud_indicators": False, "severity": 3, "coverage_category": "auto_liability"}),
    ({"policy": {"type": "home", "inception": "2018-04-22", "deductible": 500}, "claim": {"date_of_loss": "2026-08-01", "description": "Ceiling stain from upstairs bathroom leak.", "estimate_usd": 650, "photos": 2}}, {"disposition": "approve", "fraud_indicators": False, "severity": 0, "coverage_category": "property_water"}),
    ({"policy": {"type": "home", "inception": "2017-02-14", "deductible": 1000, "exclusions": ["gradual seepage", "flood (no endorsement)"]}, "claim": {"date_of_loss": "2026-07-19", "description": "Basement flooded after river overflowed during storm.", "estimate_usd": 22000}}, {"disposition": "deny", "fraud_indicators": False, "severity": 2, "coverage_category": "property_water"}),
    ({"policy": {"type": "auto", "inception": "2023-05-05", "deductible": 500}, "claim": {"date_of_loss": "2026-09-02", "description": "Hit a deer on rural road; front-end damage.", "estimate_usd": None, "photos": 0, "police_report": False}}, {"disposition": "request_info", "fraud_indicators": False, "severity": 1, "coverage_category": "auto_collision"}),
    ({"policy": {"type": "auto", "inception": "2026-08-20", "deductible": 250}, "claim": {"date_of_loss": "2026-08-24", "description": "Vehicle stolen from driveway, later found burned.", "estimate_usd": 41000, "keys_accounted_for": False, "finance_status": "two payments overdue"}}, {"disposition": "escalate", "fraud_indicators": True, "severity": 3, "coverage_category": "theft"}),
    ({"policy": {"type": "home", "inception": "2012-10-10", "deductible": 1000}, "claim": {"date_of_loss": "2026-06-05", "description": "Kitchen grease fire; range hood and cabinets damaged.", "estimate_usd": 6400, "fire_dept_report": True, "photos": 9}}, {"disposition": "approve", "fraud_indicators": False, "severity": 1, "coverage_category": "property_fire"}),
    ({"policy": {"type": "home", "inception": "2020-01-30", "deductible": 1000}, "claim": {"date_of_loss": "2026-05-12", "description": "Bicycle stolen from locked shed.", "estimate_usd": 900, "police_report": True, "receipt": True}}, {"disposition": "approve", "fraud_indicators": False, "severity": 0, "coverage_category": "theft"}),
    ({"policy": {"type": "auto", "inception": "2019-07-07", "deductible": 500}, "claim": {"date_of_loss": "2026-09-14", "description": "Multi-car pileup on highway; insured's car totalled, insured hospitalised overnight.", "estimate_usd": 36000, "police_report": True, "injuries": "insured, concussion"}}, {"disposition": "escalate", "fraud_indicators": False, "severity": 3, "coverage_category": "auto_collision"}),
    ({"policy": {"type": "home", "inception": "2016-03-03", "deductible": 1000}, "claim": {"date_of_loss": "2026-04-18", "description": "Water heater leaked; claimant states damage noticed 'a few months ago', invoices dated before policy renewal changed deductible.", "estimate_usd": 12500, "invoice_dates": ["2025-12-01", "2026-04-20"]}}, {"disposition": "request_info", "fraud_indicators": True, "severity": 2, "coverage_category": "property_water"}),
    ({"policy": {"type": "auto", "inception": "2022-12-12", "deductible": 1000}, "claim": {"date_of_loss": "2026-08-08", "description": "Insured backed into neighbour's fence.", "estimate_usd": 1400, "photos": 3, "third_party_contact": True}}, {"disposition": "approve", "fraud_indicators": False, "severity": 1, "coverage_category": "auto_liability"}),
    ({"policy": {"type": "home", "inception": "2024-02-02", "deductible": 500}, "claim": {"date_of_loss": "2026-07-04", "description": "Fireworks set deck alight; deck and siding damaged.", "estimate_usd": None, "photos": 4, "fire_dept_report": "pending"}}, {"disposition": "request_info", "fraud_indicators": False, "severity": 2, "coverage_category": "property_fire"}),
    ({"policy": {"type": "auto", "inception": "2018-08-18", "deductible": 500, "business_use": False}, "claim": {"date_of_loss": "2026-06-30", "description": "Collision while making food deliveries for a gig app.", "estimate_usd": 5200}}, {"disposition": "deny", "fraud_indicators": False, "severity": 1, "coverage_category": "auto_collision"}),
    ({"policy": {"type": "home", "inception": "2026-08-01", "deductible": 500}, "claim": {"date_of_loss": "2026-08-09", "description": "Contents fire in rented-out unit; tenant unreachable; contents list includes high-value electronics with no receipts.", "estimate_usd": 58000, "prior_claims_3y": 3}}, {"disposition": "escalate", "fraud_indicators": True, "severity": 4, "coverage_category": "property_fire"}),
    ({"policy": {"type": "auto", "inception": "2021-04-04", "deductible": 250}, "claim": {"date_of_loss": "2026-09-20", "description": "Windscreen cracked by road debris.", "estimate_usd": 420, "photos": 1}}, {"disposition": "approve", "fraud_indicators": False, "severity": 0, "coverage_category": "auto_collision"}),
    ({"policy": {"type": "home", "inception": "2014-11-11", "deductible": 1000}, "claim": {"date_of_loss": "2026-03-03", "description": "Catalytic converter stolen from car in driveway; homeowner filed under home policy.", "estimate_usd": 2100, "police_report": True}}, {"disposition": "deny", "fraud_indicators": False, "severity": 1, "coverage_category": "theft"}),
]

CMP_Q = {
    "aml_disposition": {
        "type": "choice",
        "instructions": "Disposition for this transaction-monitoring alert.",
        "criteria": {
            "close_false_positive": "Activity is explained by known customer profile; close with no further action.",
            "close_with_note": "Unusual but adequately explained; close and add a note to the customer file.",
            "escalate_l2": "Unexplained or structured activity; send to a level-2 investigator.",
            "file_sar": "Clear indicators of money laundering or terrorist financing; recommend filing a suspicious activity report.",
        },
    },
    "sanctions_hit": {
        "type": "noul",
        "instructions": "Is any party a probable true match to a sanctions or watch list entry (not merely a similar name)?",
    },
    "policy_violation": {
        "type": "choice",
        "instructions": "Which internal policy, if any, does this activity violate?",
        "criteria": {
            "none": "No policy violation.",
            "kyc_incomplete": "Customer due diligence / KYC records missing or expired.",
            "structuring": "Transactions appear split to avoid reporting thresholds.",
            "high_risk_jurisdiction": "Activity with a prohibited or high-risk jurisdiction without approval.",
            "third_party_payment": "Payments to or from unrelated third parties not allowed by the account type.",
        },
    },
    "risk_score": {
        "type": "score",
        "instructions": "Overall residual risk of this alert.",
        "criteria": ["Very low", "Low", "Medium", "High", "Critical"],
    },
}

CMP_CASES = [
    ({"customer": {"type": "individual", "occupation": "nurse", "tenure_years": 8, "kyc_status": "current"}, "alert": {"rule": "large_cash_deposit", "transactions": [{"type": "cash_in", "usd": 9800, "date": "2026-08-01"}], "note": "Customer sold a car privately; bill of sale on file."}, "screening": {"matches": []}}, {"aml_disposition": "close_with_note", "sanctions_hit": False, "policy_violation": "none", "risk_score": 1}),
    ({"customer": {"type": "individual", "occupation": "student", "tenure_years": 1, "kyc_status": "current"}, "alert": {"rule": "structuring", "transactions": [{"type": "cash_in", "usd": 9500, "date": "2026-08-02"}, {"type": "cash_in", "usd": 9400, "date": "2026-08-03"}, {"type": "cash_in", "usd": 9700, "date": "2026-08-05"}], "note": "No explanation given."}, "screening": {"matches": []}}, {"aml_disposition": "escalate_l2", "sanctions_hit": False, "policy_violation": "structuring", "risk_score": 3}),
    ({"customer": {"type": "business", "industry": "import/export", "tenure_years": 3, "kyc_status": "current"}, "alert": {"rule": "sanctions_screening", "counterparty": "Name matches SDN entry on name, date of birth and passport number."}, "screening": {"matches": [{"list": "OFAC SDN", "score": 0.98, "fields_matched": ["name", "dob", "passport"]}]}}, {"aml_disposition": "file_sar", "sanctions_hit": True, "policy_violation": "high_risk_jurisdiction", "risk_score": 4}),
    ({"customer": {"type": "individual", "occupation": "software engineer", "tenure_years": 5, "kyc_status": "current"}, "alert": {"rule": "sanctions_screening", "counterparty": "Common name; DOB differs by 31 years; different nationality."}, "screening": {"matches": [{"list": "EU consolidated", "score": 0.81, "fields_matched": ["name"]}]}}, {"aml_disposition": "close_false_positive", "sanctions_hit": False, "policy_violation": "none", "risk_score": 0}),
    ({"customer": {"type": "business", "industry": "restaurant", "tenure_years": 6, "kyc_status": "expired 2025-11-30"}, "alert": {"rule": "kyc_refresh_overdue", "transactions": [{"type": "card_settlement", "usd": 42000, "date": "2026-08-31"}]}, "screening": {"matches": []}}, {"aml_disposition": "close_with_note", "sanctions_hit": False, "policy_violation": "kyc_incomplete", "risk_score": 1}),
    ({"customer": {"type": "individual", "occupation": "retired", "tenure_years": 20, "kyc_status": "current"}, "alert": {"rule": "rapid_movement", "transactions": [{"type": "wire_in", "usd": 75000, "from": "unknown individual"}, {"type": "wire_out", "usd": 74000, "to": "crypto exchange", "hours_later": 3}], "note": "Customer says an online friend asked them to help move funds."}, "screening": {"matches": []}}, {"aml_disposition": "file_sar", "sanctions_hit": False, "policy_violation": "third_party_payment", "risk_score": 4}),
    ({"customer": {"type": "business", "industry": "payroll services", "tenure_years": 9, "kyc_status": "current"}, "alert": {"rule": "volume_spike", "transactions": [{"type": "ach_out", "usd": 480000, "count": 310}], "note": "Month-end payroll for new client onboarded last week; contract on file."}, "screening": {"matches": []}}, {"aml_disposition": "close_false_positive", "sanctions_hit": False, "policy_violation": "none", "risk_score": 0}),
    ({"customer": {"type": "business", "industry": "electronics wholesale", "tenure_years": 2, "kyc_status": "current"}, "alert": {"rule": "high_risk_geo", "transactions": [{"type": "wire_out", "usd": 120000, "to_country": "embargoed jurisdiction (synthetic)"}], "note": "No export licence on file."}, "screening": {"matches": []}}, {"aml_disposition": "escalate_l2", "sanctions_hit": False, "policy_violation": "high_risk_jurisdiction", "risk_score": 4}),
    ({"customer": {"type": "individual", "occupation": "contractor", "tenure_years": 4, "kyc_status": "current"}, "alert": {"rule": "third_party_deposits", "transactions": [{"type": "cheque_in", "usd": 3000, "payer": "unrelated individual"}, {"type": "cheque_in", "usd": 2800, "payer": "another unrelated individual"}], "note": "Customer states these are client payments for renovation jobs; invoices supplied."}, "screening": {"matches": []}}, {"aml_disposition": "close_with_note", "sanctions_hit": False, "policy_violation": "none", "risk_score": 1}),
    ({"customer": {"type": "individual", "occupation": "unemployed", "tenure_years": 0.3, "kyc_status": "incomplete: no proof of address"}, "alert": {"rule": "funnel_account", "transactions": [{"type": "cash_in", "usd": 2000, "branches": 7, "days": 4}, {"type": "atm_out_abroad", "usd": 13500}]}, "screening": {"matches": []}}, {"aml_disposition": "escalate_l2", "sanctions_hit": False, "policy_violation": "kyc_incomplete", "risk_score": 3}),
    ({"customer": {"type": "business", "industry": "charity", "tenure_years": 5, "kyc_status": "current"}, "alert": {"rule": "sanctions_screening", "counterparty": "Beneficiary bank owned by a listed entity; matches on registration number."}, "screening": {"matches": [{"list": "UK HMT", "score": 0.95, "fields_matched": ["name", "registration_no"]}]}}, {"aml_disposition": "file_sar", "sanctions_hit": True, "policy_violation": "high_risk_jurisdiction", "risk_score": 4}),
    ({"customer": {"type": "individual", "occupation": "teacher", "tenure_years": 12, "kyc_status": "current"}, "alert": {"rule": "large_wire_in", "transactions": [{"type": "wire_in", "usd": 210000, "from": "title company"}], "note": "House sale; closing statement provided."}, "screening": {"matches": []}}, {"aml_disposition": "close_false_positive", "sanctions_hit": False, "policy_violation": "none", "risk_score": 0}),
    ({"customer": {"type": "business", "industry": "cash-intensive retail", "tenure_years": 1, "kyc_status": "current"}, "alert": {"rule": "structuring", "transactions": [{"type": "cash_in", "usd": 9900, "days": 1}, {"type": "cash_in", "usd": 9900, "days": 2}, {"type": "cash_in", "usd": 9900, "days": 3}, {"type": "cash_in", "usd": 9900, "days": 4}], "note": "Sales reported to the bank are ~$2,000/day."}, "screening": {"matches": []}}, {"aml_disposition": "file_sar", "sanctions_hit": False, "policy_violation": "structuring", "risk_score": 4}),
    ({"customer": {"type": "individual", "occupation": "physician", "tenure_years": 7, "kyc_status": "current"}, "alert": {"rule": "pep_screening", "counterparty": "Customer's spouse is a local councillor (domestic PEP); declared at onboarding."}, "screening": {"matches": [{"list": "PEP", "score": 0.99, "fields_matched": ["name", "dob"]}]}}, {"aml_disposition": "close_with_note", "sanctions_hit": False, "policy_violation": "none", "risk_score": 2}),
    ({"customer": {"type": "business", "industry": "consulting", "tenure_years": 2, "kyc_status": "current", "account_type": "sole-proprietor, no third-party payments"}, "alert": {"rule": "third_party_payments", "transactions": [{"type": "wire_out", "usd": 15000, "to": "unrelated overseas individual", "count": 6}], "note": "Customer did not respond to RFI."}, "screening": {"matches": []}}, {"aml_disposition": "escalate_l2", "sanctions_hit": False, "policy_violation": "third_party_payment", "risk_score": 3}),
    ({"customer": {"type": "individual", "occupation": "pilot", "tenure_years": 10, "kyc_status": "current"}, "alert": {"rule": "foreign_atm", "transactions": [{"type": "atm_out_abroad", "usd": 600, "countries": 3}], "note": "Travel pattern consistent with occupation."}, "screening": {"matches": []}}, {"aml_disposition": "close_false_positive", "sanctions_hit": False, "policy_violation": "none", "risk_score": 0}),
    ({"customer": {"type": "business", "industry": "shipping", "tenure_years": 4, "kyc_status": "current"}, "alert": {"rule": "sanctions_screening", "counterparty": "Vessel name partially matches a listed vessel; IMO number differs."}, "screening": {"matches": [{"list": "OFAC SDN (vessels)", "score": 0.72, "fields_matched": ["vessel_name"]}]}}, {"aml_disposition": "close_with_note", "sanctions_hit": False, "policy_violation": "none", "risk_score": 1}),
    ({"customer": {"type": "individual", "occupation": "real-estate agent", "tenure_years": 3, "kyc_status": "expired 2026-01-15"}, "alert": {"rule": "cash_for_property", "transactions": [{"type": "cash_in", "usd": 48000, "days": 10}], "note": "Says deposits are client escrow funds; no escrow agreement supplied."}, "screening": {"matches": []}}, {"aml_disposition": "escalate_l2", "sanctions_hit": False, "policy_violation": "kyc_incomplete", "risk_score": 3}),
    ({"customer": {"type": "individual", "occupation": "graphic designer", "tenure_years": 2, "kyc_status": "current"}, "alert": {"rule": "p2p_volume", "transactions": [{"type": "p2p_in", "usd": 4200, "count": 35, "memo": "commission work"}]}, "screening": {"matches": []}}, {"aml_disposition": "close_false_positive", "sanctions_hit": False, "policy_violation": "none", "risk_score": 1}),
    ({"customer": {"type": "business", "industry": "money services business (unregistered)", "tenure_years": 0.5, "kyc_status": "current"}, "alert": {"rule": "unlicensed_msb", "transactions": [{"type": "wire_out", "usd": 300000, "count": 140, "to_countries": 9}]}, "screening": {"matches": []}}, {"aml_disposition": "file_sar", "sanctions_hit": False, "policy_violation": "third_party_payment", "risk_score": 4}),
]


def write(name: str, questions: dict, cases: list) -> None:
    out = HERE / name
    (out / "schema.json").write_text(json.dumps({"questions": questions}, indent=2) + "\n")
    rows = []
    for i, (state, labels) in enumerate(cases):
        rows.append({"id": f"{name}-synthetic-{i + 1:03d}", "synthetic": True, "state": state, "questions": questions, "labels": labels})
    random.Random(0).shuffle(rows)
    (out / "train.synthetic.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:15]))
    (out / "eval.synthetic.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[15:]))


if __name__ == "__main__":
    write("insurance-claims", INS_Q, INS_CASES)
    write("compliance", CMP_Q, CMP_CASES)
    print("ok")
