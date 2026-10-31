"""Language preferences from the caller text, without evaluation labels."""
import re


def language_of(text: str, previous: str) -> str:
    if re.search(r"(?:in|speak|reply in) english|angrezi|अंग्रेज़ी|अंग्रेजी", text, re.I):
        return "en"
    if re.search(r"hindi mein|हिंदी में|हिन्दी में", text, re.I):
        return "hi" if re.search(r"[\u0900-\u097f]", text) else "hi-en"
    if re.search(r"[\u0900-\u097f]", text):
        return "hi"
    if re.search(r"\b(mujhe|mera|meri|maine|hai|hain|ka|ki|kya|kab|kitne|aayega|sawaal|theek|achha|ise|kar|nahi|teesri)\b", text, re.I):
        return "hi-en"
    if re.search(r"\b(i|my|the|how|what|can|please|hello|hi|thanks|order|refund|cancel)\b", text, re.I):
        return "en"
    return previous

