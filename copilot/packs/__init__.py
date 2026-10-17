"""Domain pack registry. Add a pack here to make it available to the CLI and API."""
from __future__ import annotations


def get_pack(name: str):
    if name == "pnc_underwriting":
        from .pnc.pack import PnCUnderwritingPack
        return PnCUnderwritingPack()
    if name == "commercial_lending":
        from .lending.pack import CommercialLendingPack
        return CommercialLendingPack()
    raise ValueError(f"Unknown pack: {name}")


AVAILABLE_PACKS = ["pnc_underwriting", "commercial_lending (scaffold)"]
