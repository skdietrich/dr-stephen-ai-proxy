from typing import List

def tier_from_score(score: float) -> str:
    if score < 6:
        return "Low"
    if score < 11:
        return "Medium"
    if score < 16:
        return "High"
    return "Critical"

def mitigation_playbook(overall_score: float) -> List[str]:
    tier = tier_from_score(float(overall_score))

    if tier == "Low":
        return [
            "Document update chain and vendor access paths; baseline contract clauses.",
            "Verify logging coverage for update and admin actions.",
            "Quarterly supplier review: SBOM roadmap, patch cadence, access review."
        ]
    if tier == "Medium":
        return [
            "Enforce signed updates + staged rollouts with rollback.",
            "Segment management/update planes; least-privilege vendor access (MFA + JIT + logging).",
            "Require SBOM (or roadmap) and minimal provenance attestations."
        ]
    if tier == "High":
        return [
            "Treat as high-risk supplier: isolate update pipeline; restrict remote admin channels.",
            "Deploy secure boot / attestation where feasible; verify measured boot baselines.",
            "Add supply assurance: dual-source planning, buffer inventory for REE-dependent assemblies.",
            "Run supplier-compromise tabletop focused on malicious updates and persistence."
        ]
    return [
        "Immediate containment: remove from critical path or isolate behind strict gateways.",
        "Executive risk acceptance required if retained; accelerated replacement/dual-source plan.",
        "Out-of-band update verification; continuous monitoring of update cadence + integrity drift.",
        "Quarterly compromise drill + incident runbook for vendor compromise and firmware persistence."
    ]

