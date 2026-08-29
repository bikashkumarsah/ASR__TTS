"""Idempotently install the syllable adapter into an existing SpeeChain checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

IMPORT_AFTER = "from speechain.tokenizer.sp import SentencePieceTokenizer\n"
IMPORT_LINE = "from speechain.tokenizer.syllable import SyllableTokenizer\n"
BRANCH_AFTER = '''        elif token_type.lower() == "sentencepiece":
            self.tokenizer = SentencePieceTokenizer(
                token_path, copy_path=self.result_path
            )
'''
BRANCH_LINE = '''        elif token_type.lower() == "syllable":
            self.tokenizer = SyllableTokenizer(token_path, copy_path=self.result_path)
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patched_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if IMPORT_LINE not in text:
        if IMPORT_AFTER not in text:
            raise RuntimeError(f"Unsupported SpeeChain import layout: {path}")
        text = text.replace(IMPORT_AFTER, IMPORT_AFTER + IMPORT_LINE, 1)
    if BRANCH_LINE not in text:
        if BRANCH_AFTER not in text:
            raise RuntimeError(f"Unsupported SpeeChain tokenizer branch layout: {path}")
        text = text.replace(BRANCH_AFTER, BRANCH_AFTER + BRANCH_LINE, 1)
    text = text.replace("['char', 'sentencepiece']", "['char', 'sentencepiece', 'syllable']")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speechain-root", required=True)
    parser.add_argument("--check", action="store_true", help="Validate compatibility without modifying the checkout")
    args = parser.parse_args()
    root = Path(args.speechain_root).expanduser().resolve()
    models = [root / "speechain" / "model" / name for name in ("ar_asr.py", "lm.py")]
    adapter_source = Path(__file__).resolve().with_name("syllable.py")
    adapter_target = root / "speechain" / "tokenizer" / "syllable.py"
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "check_only": args.check, "files": {}}
    replacements = {path: patched_text(path) for path in models}
    if not args.check:
        for path, text in replacements.items():
            before = sha256(path)
            backup = path.with_suffix(path.suffix + ".synthetic-asr.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(text, encoding="utf-8")
            report["files"][str(path)] = {"before": before, "after": sha256(path), "backup": str(backup)}
        adapter_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter_source, adapter_target)
        report["files"][str(adapter_target)] = {"after": sha256(adapter_target)}
    else:
        for path, text in replacements.items():
            report["files"][str(path)] = {"current": sha256(path), "compatible": bool(text)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
