from typing import Dict, Any

def _boolish(val: Any) -> str:
    if val is None:
        return "unknown"
    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return "yes"
    if s in {"false", "0", "no", "n"}:
        return "no"
    return "unknown"

def score_ree_concentration(row: Dict[str, Any]) -> float:
    contains = _boolish(row.get("contains_ree_magnets"))
    if contains == "no":
        return 0.0

    score = 6.0
    alt = _boolish(row.get("alt_supplier_available"))
    spof = _boolish(row.get("known_single_point_of_failure"))

    if alt == "no":
        score += 5.0
    elif alt == "unknown":
        score += 2.0

    if spof == "yes":
        score += 5.0
    elif spof == "unknown":
        score += 2.0

    juris = (row.get("origin_jurisdiction") or "").strip().lower()
    if any(x in juris for x in ["strategic competitor", "prc", "china"]):
        score += 3.0
    elif juris:
        score += 1.0
    else:
        score += 1.5

    try:
        credit = float(row.get("ree_supply_mitigation_credit") or 0.0)
    except Exception:
        credit = 0.0
    score -= max(0.0, min(4.0, credit))

    return max(0.0, min(20.0, score))

def score_firmware_integrity(row: Dict[str, Any]) -> float:
    score = 0.0

    ota = _boolish(row.get("firmware_ota"))
    signing = _boolish(row.get("firmware_signing"))
    att = _boolish(row.get("secure_boot_attestation"))
    sbom = _boolish(row.get("sbom_available"))
    remote = _boolish(row.get("remote_admin_access"))
    logging = _boolish(row.get("telemetry_logging"))

    if ota == "yes":
        score += 5.0
    elif ota == "unknown":
        score += 2.5

    if remote == "yes":
        score += 3.0
    elif remote == "unknown":
        score += 1.5

    if signing == "no":
        score += 6.0
    elif signing == "unknown":
        score += 3.0

    if att == "no":
        score += 4.0
    elif att == "unknown":
        score += 2.0

    if sbom == "no":
        score += 2.0
    elif sbom == "unknown":
        score += 1.0

    if logging == "no":
        score += 2.0
    elif logging == "unknown":
        score += 1.0

    return max(0.0, min(20.0, score))

def score_overall(row: Dict[str, Any], weight_fw: float = 0.55) -> Dict[str, float]:
    ree = score_ree_concentration(row)
    fw = score_firmware_integrity(row)
    weight_fw = max(0.0, min(1.0, float(weight_fw)))
    weight_ree = 1.0 - weight_fw
    overall = weight_ree * ree + weight_fw * fw
    return {"ree_risk": ree, "firmware_risk": fw, "overall_risk": overall}
