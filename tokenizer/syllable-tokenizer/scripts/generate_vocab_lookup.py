from pathlib import Path

# Define the characters to be included in the vocabulary lookup
CONSONANTS = [
    'क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण','त','थ','द','ध','न','प','फ','ब','व','भ','म','य','र','ल','व','श','ष','स','ह','क्ष','त्र','ज्ञ','त्त','द्ध','श्र','द्य'
]
VOWEL_SIGNS = ['ा','ि', 'ी', 'ु','ू','े','ै', 'ो','ौ','ं','्']
INDEPENDENT_VOWELS = ['अ','आ','ई','इ','उ','ऊ','ए','ऐ','ॐ','ॠ', 'ऋ','ऌ']
POSTFIX_NASAL = 'ँ'
REPHA = '्र'
HALANT = '्'
EXTRA_CHARS = [' ','।','?','!']
NUMBERS = ['०','१','२','३','४','५','६','७','८','९']


def build_lookup_tokens():
    tokens = []

    for consonant in CONSONANTS:
        tokens.append(consonant)
        tokens.append(consonant + HALANT)
        tokens.append(consonant + POSTFIX_NASAL)
        tokens.append(consonant + REPHA)
        tokens.append(consonant + REPHA + POSTFIX_NASAL)

        for vowel_sign in VOWEL_SIGNS:
            tokens.append(consonant + vowel_sign)

            if vowel_sign != HALANT:
                tokens.append(consonant + vowel_sign + POSTFIX_NASAL)
                tokens.append(consonant + REPHA + vowel_sign)
                tokens.append(consonant + REPHA + vowel_sign + POSTFIX_NASAL)

    tokens.extend(INDEPENDENT_VOWELS)
    tokens.extend(EXTRA_CHARS)
    tokens.extend(NUMBERS)

    return list(dict.fromkeys(tokens))


if __name__ == "__main__":
    tokens = build_lookup_tokens()
    print(f"Total tokens Generated: {len(tokens)}")

    output_file = Path(__file__).resolve().parents[1] / "dataset" / "nepali_syllables_lookup.vocab"
    output_file.write_text("\n".join(tokens), encoding="utf-8")

    print(f"Lookup tokens saved to {output_file}")
