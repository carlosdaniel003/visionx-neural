"""Preparação V3 de máscaras de corpo inteiro DESLOCADO.

Etapa A — visualizar 34 pares (quantidade depende do arquivo atual):
    python -m src.scripts.prepare_deslocado_body_masks_v3

Etapa B — após ajustar e aprovar todas as boxes no JSON:
    python -m src.scripts.prepare_deslocado_body_masks_v3 --review "...\body_masks_review.json"

Não treina CNN nem altera produção. As caixas iniciais são sugestões.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.services.deslocado_body_masks_v3 import (
    prepare_body_masks, validate_body_masks,
)
from src.services.deslocado_neural_dataset import latest_deslocado_manifest


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(
        description="Preparar/analisar máscara geométrica DESLOCADO com revisão humana"
    )
    parser.add_argument(
        "--root",type=Path,default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--manifest",type=Path,default=None)
    parser.add_argument(
        "--review",type=Path,default=None,
        help="Validar arquivo body_masks_review.json editado pelo operador",
    )
    args=parser.parse_args(argv)
    if args.review:
        result,out=validate_body_masks(args.review)
        print(
            "Máscaras validadas (somente geométricas):",
            result["total_approved"]
        )
        print("CATÁLOGO:",out/"validated_body_masks.json")
        print("PREVIEWS VERDES:",out/"preview_validated")
    else:
        manifest=args.manifest or latest_deslocado_manifest(args.root)
        result,out=prepare_body_masks(manifest)
        print("Imagens a revisar:",result["total_images"])
        print("Por luz:",result["lighting_distribution"])
        print("REVISÃO JSON:",out/"body_masks_review.json")
        print("PREVIEWS LARANJAS:",out/"preview_proposals")
        print("Atenção: nenhuma caixa está aprovada automaticamente.")
    print("Nenhum treinamento ou motor operacional foi alterado.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
