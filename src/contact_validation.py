"""Shared validation for customer name, email, and phone (onboarding + tools)."""


def phone_digits(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())


def validate_customer_profile(full_name: str, email: str, phone: str) -> list[str]:
    """
    Return human-readable errors; empty means valid.

    Full name and phone are required. Email is optional: if omitted or blank, no email
    checks run; if provided, it must look like a normal address (@ and domain with a dot).
    """
    errors: list[str] = []
    name = (full_name or "").strip()
    if len(name) < 2:
        errors.append("Full name must be at least 2 characters.")
    em = (email or "").strip()
    if em:
        if "@" not in em:
            errors.append("Email must look valid (include @ and a domain with a dot).")
        else:
            local, _, domain = em.partition("@")
            if not local or not domain or "." not in domain:
                errors.append("Email must look valid (include @ and a domain with a dot).")
    digits = phone_digits(phone)
    if len(digits) < 10:
        errors.append("Phone must contain at least 10 digits (area code + number).")
    return errors
