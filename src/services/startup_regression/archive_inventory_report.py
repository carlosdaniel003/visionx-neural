"""Relatórios locais do inventário — fora dos arquivos OK/NG e dataset."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def human_summary(report: dict) -> str:
    summary = report.get("summary", {})
    lines = [
        "ODIN | INVENTÁRIO DE EVIDÊNCIAS OK/NG | ETAPA 1",
        "=" * 58,
        "SOMENTE DIAGNÓSTICO — NÃO EXECUTA ANÁLISE DE DEFEITOS.",
        "NÃO É UM GATE DE INICIALIZAÇÃO.",
        "",
        f"Raiz: {report.get('root', '')}",
        f"Gerado (UTC): {report.get('generated_at_utc', '')}",
        f"Estado do inventário: {report.get('status', '')}",
        "",
        f"PNG descobertos: {summary.get('png_count', 0)}",
        f"PNG legíveis: {summary.get('valid_png', 0)}",
        f"PNG inválidos: {summary.get('invalid_png', 0)}",
        f"Legado SIDE (inferido): {summary.get('legacy_side_count', 0)}",
        f"Iluminações explícitas sem manifesto: {summary.get('explicit_unlinked_count', 0)}",
        f"PNG vinculados por manifesto: {summary.get('manifest_linked_png_count', 0)}",
        f"Eventos com manifesto válido: {summary.get('linked_event_count', 0)}",
        f"Grupos de imagens repetidas: {summary.get('pixel_duplicate_groups', 0)}",
        f"Conflitos de pixels OK x NG: {summary.get('cross_label_conflict_groups', 0)}",
        f"Pendências de qualificação: {summary.get('issue_count', 0)}",
        "",
    ]
    for title, key in [
        ("POR RÓTULO", "by_label"),
        ("POR ILUMINAÇÃO (DECLARADA/INFERIDA)", "by_lighting"),
        ("CATEGORIAS SUGERIDAS PELO NOME", "by_category_hint"),
    ]:
        lines.append(title)
        values = summary.get(key, {})
        for name, count in sorted(values.items()):
            lines.append(f"  {name}: {count}")
        if not values:
            lines.append("  Nenhum registro")
        lines.append("")

    lines.extend([
        "PENDÊNCIAS (não alteram arquivos):",
    ])
    counted = Counter(item.get("code", "UNKNOWN") for item in report.get("issues", []))
    if counted:
        for code, count in sorted(counted.items()):
            lines.append(f"  {code}: {count}")
    else:
        lines.append("  Nenhuma pendência estrutural")
    lines.append("")
    for issue in report.get("issues", []):
        lines.append(
            f"- [{issue.get('code')}] {issue.get('path')}: {issue.get('detail')}"
        )
    lines.extend([
        "",
        "IMPORTANTE: A detecção das barras, OCR, especialistas, KNN e fusão",
        "multilight NÃO foram executados nesta Etapa 1.",
        "Rótulos de arquivo OK/NG não constituem acertos da IA.",
        "Somente um manifesto real autoriza agrupar SIDE/TOP/MID de uma peça.",
    ])
    return "\n".join(lines) + "\n"


def _atomic_text(target: Path, value: str) -> None:
    """Grava relatório de forma atômica; nunca altera arquivos do acervo."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix="._inventory_",
            suffix=".tmp",
            delete=False,
        ) as temp:
            temp_name = temp.name
            temp.write(value)
        os.replace(temp_name, target)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def write_reports(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output = Path(output_dir).resolve()
    root = Path(report["root"]).resolve()
    for label in ("ng_archive", "ok_archive"):
        archive = (root / "public" / label).resolve()
        if output == archive or archive in output.parents:
            raise ValueError("Relatórios não podem ser escritos dentro do acervo")
    dataset = (root / "public" / "dataset").resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("Relatórios não podem ser escritos dentro do dataset")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = output / f"inventory_{stamp}.json"
    text_path = output / f"inventory_{stamp}.txt"
    _atomic_text(
        json_path,
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_text(text_path, human_summary(report))
    return json_path, text_path


__all__ = ["human_summary", "write_reports"]
