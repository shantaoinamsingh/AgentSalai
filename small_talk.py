#!/usr/bin/env python3
"""Instant replies for pure small talk, so a "hello" never costs a model call.

Only messages made up *entirely* of small-talk phrases match: "hi, can you
summarise this?" still goes to the model. Bare acknowledgements ("ok", "yes")
are deliberately excluded because they are often answers to a question the
assistant just asked.
"""
import re
from typing import List, Optional

PHRASES = {
    "bye": [
        "goodbye", "good bye", "bye bye", "bye", "see you later", "see you soon", "see you",
        "see ya", "cya", "good night", "goodnight", "take care", "talk to you later", "ttyl",
    ],
    "thanks": [
        "thank you so much", "thank you very much", "thanks a lot", "thanks so much",
        "thank you", "many thanks", "much appreciated", "appreciate it", "thanks", "thank u",
        "thx", "ty", "cheers",
    ],
    "identity": [
        "who are you", "what are you", "what is your name", "whats your name", "what can you do",
        "what do you do", "how can you help", "how can you help me", "what can you help with",
        "help me", "help",
    ],
    "how_are_you": [
        "how are you doing", "how are you", "how r u", "how are u", "hows it going",
        "how is it going", "how do you do", "whats up", "wassup", "sup",
    ],
    "greeting": [
        "good morning", "good afternoon", "good evening", "good day", "hello", "hiya", "hi",
        "hey", "heya", "howdy", "greetings", "hola", "namaste", "yo",
    ],
}

# Words that may accompany a phrase without changing its meaning.
FILLER = {"there", "salai", "bot", "again", "all", "everyone", "friend", "buddy", "and", "so", "very", "much", "oh"}

# Precedence when a message mixes categories ("thanks, bye" -> bye).
ORDER = ["bye", "thanks", "identity", "how_are_you", "greeting"]

_ALTERNATIVES = sorted(
    ((p, cat) for cat, ps in PHRASES.items() for p in ps), key=lambda pc: -len(pc[0])
)


def _normalise(text: str) -> str:
    text = text.lower().replace("’", "'").replace("'", "")
    text = re.sub(r"[^a-z\s]", " ", text)
    # Stretched greetings: "hiii", "heyyy", "hellooo".
    text = re.sub(r"\b(h+i+|he+y+|hel+o+)\b", lambda m: _unstretch(m.group(0)), text)
    return " ".join(text.split())


def _unstretch(word: str) -> str:
    if re.fullmatch(r"h+i+", word):
        return "hi"
    if re.fullmatch(r"he+y+", word):
        return "hey"
    return "hello"


def _categories(text: str) -> Optional[List[str]]:
    """Categories found if the whole message is small talk, else None."""
    words = _normalise(text).split()
    if not words or len(words) > 12:
        return None
    found: List[str] = []
    i = 0
    while i < len(words):
        if words[i] in FILLER:
            i += 1
            continue
        for phrase, cat in _ALTERNATIVES:
            n = len(phrase.split())
            if words[i : i + n] == phrase.split():
                found.append(cat)
                i += n
                break
        else:
            return None
    return found or None


def reply_for(text: str, can_browse: bool = False) -> Optional[str]:
    """A canned reply for pure small talk, or None to let the model answer."""
    cats = _categories(text)
    if not cats:
        return None
    top = min(cats, key=ORDER.index)

    if top == "bye":
        return "Goodbye! Come back any time you need a hand."
    if top == "thanks":
        return "You're welcome! Anything else I can help with?"
    if top == "identity":
        abilities = [
            "answer questions and explain things",
            "write, summarise and reformat text (tables, reports, charts)",
            "work with files and screenshots you attach or paste into the chat",
        ]
        if can_browse:
            abilities.append("open websites in a browser to look things up for you")
        return "I'm Salai, your AI assistant. I can:\n\n" + "\n".join(f"- {a}" for a in abilities) + (
            "\n\nJust ask in your own words."
        )

    greeting = _greeting_word(text) if "greeting" in cats else "Hi"
    if "how_are_you" in cats:
        return f"{greeting}! I'm doing well, thanks for asking. What can I help you with today?"
    return f"{greeting}! How can I help you today?"


def _greeting_word(text: str) -> str:
    norm = _normalise(text)
    for tod in ("good morning", "good afternoon", "good evening"):
        if tod in norm:
            return tod.capitalize()
    return "Hello"
