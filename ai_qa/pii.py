"""
Deterministic PII check: finds email addresses and Swedish mobile numbers
in a text with regular expressions. No LLM, no cost, same result every run.
"""
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?:\+46\s?|0)7\d[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}")


def find_pii(text):
    return EMAIL.findall(text) + PHONE.findall(text)