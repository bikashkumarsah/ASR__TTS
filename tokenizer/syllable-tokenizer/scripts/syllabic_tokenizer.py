#!/usr/bin/env python3
"""Utilities for syllabic tokenization and text cleanup."""

from __future__ import annotations

import html
import re


_HTML_TAG_RE = re.compile(r"<[^>]*>")
_NON_DEVA_RE = re.compile(r"[^\u0900-\u097F\s]")
_SPACE_RE = re.compile(r"\s+")
_MAX_TOKEN_LENGTH_CACHE = {}

# The pronunciation-aware tokenizer used by Ghimire et al. evaluates lookup
# candidates in a fixed window of at most four Unicode code points.  Keep that
# paper-comparable behaviour as the project default.  Passing
# ``max_token_length=None`` to ``tokenize`` remains an explicit opt-in to the
# former full-vocabulary (currently seven-code-point) search for reproducibility
# of already completed runs.
DEFAULT_LOOKUP_WINDOW_SIZE = 4


def clean_text(text):
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _NON_DEVA_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def print_debug(data, IS_DEBUG):
    if IS_DEBUG:
        print(data)


def get_lookup_tokens(lookup_vocab_file="dataset/nepali_syllables_lookup.vocab"):
    with open(lookup_vocab_file, 'r', encoding='utf-8') as f:
        tokens_file = f.read()
    syllabic_tokens = frozenset(token for token in tokens_file.split('\n') if token)
    return syllabic_tokens


def _max_token_length(lookup_vocab):
    """Cache the longest lookup entry for repeated sentence tokenization."""
    key = id(lookup_vocab)
    cached = _MAX_TOKEN_LENGTH_CACHE.get(key)
    if cached is not None and cached[0] is lookup_vocab:
        return cached[1]
    value = max((len(token) for token in lookup_vocab), default=1)
    _MAX_TOKEN_LENGTH_CACHE[key] = (lookup_vocab, value)
    return value


def tokenize(
    sentense,
    lookup_vocab,
    debug=False,
    max_token_length=DEFAULT_LOOKUP_WINDOW_SIZE,
):
    """Tokenize with greedy longest lookup matching.

    The default four-code-point window matches the published
    pronunciation-aware tokenizer.  Use ``max_token_length=None`` only when
    reproducing the legacy full-vocabulary-window results.
    """
    if max_token_length is not None and max_token_length < 1:
        raise ValueError("max_token_length must be a positive integer or None")
    sentense = clean_text(sentense)
    aligned_tokens = [ ]
    pos = 0
    print_debug(f"Total length of Sent : {len(sentense)}", debug)
    while (pos < len(sentense)):
        print_debug(f"current pos : {pos}", debug)
        move_upto = (
            _max_token_length(lookup_vocab)
            if max_token_length is None
            else max_token_length
        )
        # if(pos == len(chars)-1): aligned_tokens.append(chars[pos])

        if pos + move_upto > (len(sentense)): move_upto = len(sentense) - pos
        while (move_upto != 0):
            interm_token = sentense[pos:pos+move_upto]
            move_upto = move_upto - 1
            if interm_token in lookup_vocab:
                aligned_tokens.append(interm_token)
                print_debug(f"interm_token {interm_token} in pos = {pos} and move_upto = {move_upto}" , debug)
                break
        #special case for the last char
        pos = pos + 1 + move_upto
    print_debug(f"Aligned Token: {aligned_tokens}", debug)
    return aligned_tokens


if __name__ == "__main__":
    sentense = "नेपाल एक सुन्दर देश हो।"
    syllabic_tokens = get_lookup_tokens()
    tokens = tokenize(sentense, syllabic_tokens, debug=False)
    print(f"Total tokens generated: {len(tokens)}")
    print(f"Tokens: {' ## '.join(tokens)}")
