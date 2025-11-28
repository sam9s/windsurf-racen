from __future__ import annotations

"""Business facts repository for RACEN.

This module centralises small, structured business facts such as
payment methods, COD availability, support contact details and
canonical policy/review URLs. It is deliberately a thin layer
over a YAML file so we can later swap the storage implementation
(e.g., to Postgres) without changing callers.
"""

from pathlib import Path
from typing import Optional

import yaml  # type: ignore
from pydantic import BaseModel


class PaymentConfig(BaseModel):
    """Payment configuration.

    Args:
        methods: List of supported payment methods.
        cod_enabled: Whether Cash on Delivery (COD) is currently available.
    """

    methods: list[str] = []
    cod_enabled: bool = False


class SupportConfig(BaseModel):
    """Support contact configuration.

    Args:
        phone: Primary support phone number.
        email: Primary support email address.
        hours_weekday: Human-readable weekday support hours.
        hours_weekend: Human-readable weekend/holiday support hours.
    """

    phone: str = ""
    email: str = ""
    hours_weekday: Optional[str] = None
    hours_weekend: Optional[str] = None


class ReturnsConfig(BaseModel):
    """Returns and cancellation configuration.

    Args:
        window_days: Number of days from delivery when returns are allowed.
    """

    window_days: int = 0


class UrlsConfig(BaseModel):
    """Canonical URLs for key policy and reputation pages.

    Args:
        contact: Contact/support page URL.
        returns: Returns and cancellation policy URL.
        shipping: Shipping policy URL.
        privacy: Privacy policy URL.
        terms: Terms and conditions URL.
        trustpilot: Trustpilot reviews URL for Grest.
        mouthshut: MouthShut reviews URL for Grest.
    """

    contact: Optional[str] = None
    returns: Optional[str] = None
    shipping: Optional[str] = None
    warranty: Optional[str] = None
    privacy: Optional[str] = None
    terms: Optional[str] = None
    trustpilot: Optional[str] = None
    mouthshut: Optional[str] = None


class BusinessFacts(BaseModel):
    """Top-level container for structured business facts."""

    payment: PaymentConfig = PaymentConfig()
    support: SupportConfig = SupportConfig()
    returns: ReturnsConfig = ReturnsConfig()
    urls: UrlsConfig = UrlsConfig()


class YamlBusinessFactsRepository:
    """YAML-backed repository for business facts.

    Args:
        yaml_path: Optional explicit path to the YAML file. When omitted,
            the path defaults to ``Grest_Data/business_facts.yaml`` under
            the project root.
    """

    def __init__(self, yaml_path: Optional[Path] = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self._path = (
            yaml_path
            or project_root / "Grest_Data" / "business_facts.yaml"
        )

    def load(self) -> BusinessFacts:
        """Load and validate business facts from YAML.

        Returns:
            BusinessFacts: Parsed business facts with defaults on error.
        """

        try:
            if not self._path.exists():
                return BusinessFacts()
            with self._path.open("r", encoding="utf-8", errors="ignore") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return BusinessFacts()

        try:
            payment_raw = data.get("payment") or {}
            support_raw = data.get("support") or {}
            returns_raw = data.get("returns") or {}
            urls_raw = data.get("urls") or {}
        except Exception:
            return BusinessFacts()

        payment = (
            PaymentConfig(**payment_raw)
            if isinstance(payment_raw, dict)
            else PaymentConfig()
        )
        support = (
            SupportConfig(**support_raw)
            if isinstance(support_raw, dict)
            else SupportConfig()
        )
        returns = (
            ReturnsConfig(**returns_raw)
            if isinstance(returns_raw, dict)
            else ReturnsConfig()
        )
        urls = (
            UrlsConfig(**urls_raw)
            if isinstance(urls_raw, dict)
            else UrlsConfig()
        )

        return BusinessFacts(payment=payment, support=support, returns=returns, urls=urls)


_DEFAULT_REPO = YamlBusinessFactsRepository()


def get_business_facts() -> BusinessFacts:
    """Return the current business facts from the default repository.

    Returns:
        BusinessFacts: Parsed business facts with safe defaults on error.
    """

    return _DEFAULT_REPO.load()
