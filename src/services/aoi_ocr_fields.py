"""Pós-processamento e segunda leitura dos campos de texto da AOI.

A leitura geral continua sendo feita pelo ScreenMonitor. Este módulo corrige
artefatos de borda e, somente quando necessário, relê o campo Parts por uma ROI
ancorada no rótulo da própria interface.
"""

from __future__ import annotations

import re

import cv2


def strip_ocr_field_noise(value: str) -> str:
    """Remove artefatos de borda/célula sem alterar o conteúdo interno."""
    text = str(value or "").strip()
    text = re.sub(r"^[\[\]{}|¦!;:'\"‘’“”]+\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_board_ocr(value: str) -> str:
    return strip_ocr_field_noise(value)


_NUMERIC_OCR_TRANSLATION = str.maketrans(
    {
        "I": "1",
        "i": "1",
        "L": "1",
        "l": "1",
        "|": "1",
        "O": "0",
        "o": "0",
        "Q": "0",
        "q": "0",
        "S": "5",
        "s": "5",
        "$": "5",
        "Z": "2",
        "z": "2",
        "G": "6",
        "g": "6",
        "B": "8",
        "b": "8",
    }
)


def _normalize_numeric_operand(value: str) -> str:
    """Normaliza um token somente quando a gramática exige número."""
    token = str(value or "").strip().translate(_NUMERIC_OCR_TRANSLATION)
    token = re.sub(r"[^0-9.,+\-]", "", token)
    token = token.replace(",", ".")
    token = re.sub(r"\.{2,}", ".", token)
    return token.strip(".")


def normalize_value_ocr(value: str) -> str:
    """Normaliza operandos numéricos de expressões comparativas da AOI."""
    text = strip_ocr_field_noise(value)
    match = re.match(
        r"^(.+?)\s*(<=|>=|<|>)\s*(.+?)\s*(<=|>=|<|>)\s*([^\s]+)(.*)$",
        text,
    )
    if not match:
        return re.sub(r"\s+", " ", text).strip()

    left = _normalize_numeric_operand(match.group(1))
    middle = _normalize_numeric_operand(match.group(3))
    right = _normalize_numeric_operand(match.group(5))
    if not left or not middle or not right:
        return re.sub(r"\s+", " ", text).strip()

    suffix = re.sub(r"\s+", " ", match.group(6)).strip()
    normalized = (
        f"{left} {match.group(2)} {middle} "
        f"{match.group(4)} {right}"
    )
    if suffix:
        normalized += f" {suffix}"
    return normalized.strip()


def normalize_parts_ocr(value: str) -> str:
    """Normaliza somente a parte numérica de uma referência de componente."""
    text = re.sub(r"\s+", "", strip_ocr_field_noise(value)).upper()
    if not text:
        return text

    first_digit = re.search(r"\d", text)
    if first_digit:
        index = first_digit.start()
        prefix = text[:index]
        suffix = text[index:].translate(_NUMERIC_OCR_TRANSLATION)
        suffix = re.sub(r"[^0-9~\-/,.]", "", suffix)
        candidate = prefix + suffix
        if candidate:
            return candidate

    # Sem dígito explícito (ex.: RI~5), a releitura dirigida decide o número.
    return text


def looks_like_component_reference(value: str) -> bool:
    text = str(value or "").strip().upper()
    return bool(
        re.fullmatch(
            r"[A-Z]{1,4}\d+(?:[~\-]\d+)?(?:[/,][A-Z]?\d+)*",
            text,
        )
    )


def _prepare_binary(image, scale: int = 3):
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(
        gray,
        None,
        fx=max(1, int(scale)),
        fy=max(1, int(scale)),
        interpolation=cv2.INTER_CUBIC,
    )
    _, binary = cv2.threshold(
        enlarged,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary


def _token_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _field_token_matches(token: str, target: str) -> bool:
    """Tolera pequenas deformações OCR no próprio rótulo da AOI."""
    token_key = _token_key(token)
    target_key = _token_key(target)
    if token_key == target_key:
        return True
    if len(token_key) < 3 or len(target_key) < 3:
        return False

    import difflib

    return difflib.SequenceMatcher(
        None,
        token_key,
        target_key,
    ).ratio() >= 0.72


def _find_field_crop(text_zone, field_name: str, pytesseract_module):
    binary = _prepare_binary(text_zone, scale=3)
    if binary is None:
        return None

    try:
        data = pytesseract_module.image_to_data(
            binary,
            config="--psm 6 --oem 3",
            output_type=pytesseract_module.Output.DICT,
        )
    except Exception:
        return None

    count = len(data.get("text", []))
    rows = {}
    for index in range(count):
        token = str(data["text"][index] or "").strip()
        if not token:
            continue
        key = (
            data.get("block_num", [0] * count)[index],
            data.get("par_num", [0] * count)[index],
            data.get("line_num", [0] * count)[index],
        )
        rows.setdefault(key, []).append(
            {
                "text": token,
                "key": _token_key(token),
                "left": int(data["left"][index]),
                "top": int(data["top"][index]),
                "width": int(data["width"][index]),
                "height": int(data["height"][index]),
            }
        )

    target = _token_key(field_name)
    stop_labels = {
        "BOARD",
        "BLOCK",
        "KIND",
        "PART",
        "PARTS",
        "PROCESS",
        "STEP",
        "TERMINAL",
        "VALUE",
    }

    candidate_rows = []
    for words in rows.values():
        words = sorted(words, key=lambda item: item["left"])
        keys = [item["key"] for item in words]
        matching_indexes = [
            index
            for index, item in enumerate(words)
            if _field_token_matches(item["text"], field_name)
        ]
        if not matching_indexes:
            continue
        if target == "PARTS" and "KIND" in keys:
            continue
        candidate_rows.append((words, matching_indexes[0]))

    if not candidate_rows:
        return None

    words, anchor_index = min(
        candidate_rows,
        key=lambda item: min(w["top"] for w in item[0]),
    )
    anchor = words[anchor_index]

    x1 = anchor["left"] + anchor["width"] + 2
    x2 = binary.shape[1]
    for word in words[anchor_index + 1:]:
        if word["key"] in stop_labels:
            x2 = max(x1 + 2, word["left"] - 4)
            break

    row_top = min(word["top"] for word in words)
    row_bottom = max(word["top"] + word["height"] for word in words)
    y_pad = max(6, int((row_bottom - row_top) * 0.35))
    y1 = max(0, row_top - y_pad)
    y2 = min(binary.shape[0], row_bottom + y_pad)

    if x2 <= x1 + 3 or y2 <= y1 + 3:
        return None
    return binary[y1:y2, x1:x2].copy()


def recover_parts_from_text_zone(
    text_zone,
    parsed_value: str,
    pytesseract_module,
) -> str:
    """Relê Parts quando a leitura geral não obedece à sintaxe de componente."""
    fallback = normalize_parts_ocr(parsed_value)
    if looks_like_component_reference(fallback):
        return fallback

    crop = _find_field_crop(text_zone, "Parts", pytesseract_module)
    if crop is None or crop.size == 0:
        return fallback

    candidates = []
    try:
        full = pytesseract_module.image_to_string(
            crop,
            config=(
                "--psm 7 --oem 3 "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                "0123456789~-_/"
            ),
        ).strip()
        if full:
            candidates.append(normalize_parts_ocr(full))
    except Exception:
        pass

    numeric = ""
    try:
        numeric = pytesseract_module.image_to_string(
            crop,
            config=(
                "--psm 7 --oem 3 "
                "-c tessedit_char_whitelist=0123456789~-"
            ),
        ).strip()
        numeric = re.sub(r"[^0-9~-]", "", numeric)
    except Exception:
        numeric = ""

    if numeric and re.fullmatch(r"\d+(?:[~-]\d+)?", numeric):
        left = fallback.split("~", 1)[0].split("-", 1)[0]
        prefix = re.match(r"[A-Za-z]{1,4}", left)
        prefix_text = prefix.group(0) if prefix else ""
        # Ex.: OCR geral RI~5, leitura numérica dirigida 3~5 -> R3~5.
        prefix_text = re.sub(r"[IiLlOo]+$", "", prefix_text)
        if prefix_text:
            candidates.append(prefix_text.upper() + numeric)

    valid = [
        candidate
        for candidate in candidates
        if looks_like_component_reference(candidate)
    ]
    if valid:
        return min(valid, key=lambda item: (len(item), item))

    return fallback


__all__ = [
    "looks_like_component_reference",
    "normalize_board_ocr",
    "normalize_parts_ocr",
    "normalize_value_ocr",
    "recover_parts_from_text_zone",
    "strip_ocr_field_noise",
]
