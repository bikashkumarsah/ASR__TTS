"""Checksum-pinned SpeeChain adapter for the project's Nepali lookup tokenizer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os

import torch
from speechain.tokenizer.abs import Tokenizer
from speechain.utilbox.import_util import parse_path_args


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SyllableTokenizer(Tokenizer):
    """Use the exact tokenizer source and lookup exported with a corpus run."""

    def tokenizer_init_fn(self, token_path: str, copy_path: str | None = None, **tokenizer_conf):
        del copy_path, tokenizer_conf
        root = parse_path_args(token_path)
        manifest_path = os.path.join(root, "tokenizer_manifest.json")
        lookup_path = os.path.join(root, "lookup.vocab")
        source_path = os.path.join(root, "syllabic_tokenizer.py")
        for path in (manifest_path, lookup_path, source_path):
            if not os.path.isfile(path):
                raise RuntimeError(f"Pinned syllable-tokenizer artifact is missing: {path}")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        checks = {
            "vocabulary_sha256": _sha256(lookup_path),
            "tokenizer_sha256": _sha256(source_path),
        }
        mismatches = {
            key: {"expected": manifest.get(key), "actual": value}
            for key, value in checks.items() if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Pinned syllable tokenizer checksum mismatch: {mismatches}")
        spec = importlib.util.spec_from_file_location("speechain_pinned_syllabic_tokenizer", source_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import pinned tokenizer: {source_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module
        self._lookup = module.get_lookup_tokens(lookup_path)

    def text2tensor(self, text: str, no_sos: bool = False, no_eos: bool = False, return_tensor: bool = True):
        normalized = self._module.clean_text(text).replace("।", " ").replace("॥", " ").replace("?", " ").replace("!", " ")
        emitted = self._module.tokenize(normalized, self._lookup)
        indices = [] if no_sos else [self.sos_eos_idx]
        for token in emitted:
            if token.isspace():
                if self.space_idx is not None:
                    indices.append(self.space_idx)
            else:
                indices.append(self.token2idx.get(token, self.unk_idx))
        if not no_eos:
            indices.append(self.sos_eos_idx)
        return torch.LongTensor(indices) if return_tensor else indices
