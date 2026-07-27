"""Especialista semântico com embedding 128D e reconstrução diagnóstica."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.distance import cosine


class SemanticExpert:
    """
    Compara embeddings densos de 128 dimensões.

    Esquema:
    - 0..15: densidade de bordas em grid 4x4;
    - 16..31: brilho médio em grid 4x4;
    - 32..63: histograma de matiz (H);
    - 64..95: histograma de saturação (S);
    - 96..127: histograma de valor/brilho (V).
    """

    SCHEMA_VERSION = "visionx.semantic.v2"
    EMBEDDING_SIZE = 128
    GROUPS = (
        ("edge_density", 0, 16, "EDGE"),
        ("brightness", 16, 32, "LUMA"),
        ("hue_histogram", 32, 64, "HUE"),
        ("saturation_histogram", 64, 96, "SAT"),
        ("value_histogram", 96, 128, "VAL"),
    )

    def __init__(self):
        self.embedding_size = self.EMBEDDING_SIZE

    @staticmethod
    def _safe_pair(
        reference: np.ndarray,
        query: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ref = reference.copy()
        test = query.copy()
        if ref.shape != test.shape:
            test = cv2.resize(
                test,
                (ref.shape[1], ref.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return ref, test

    @staticmethod
    def _extract_focus(
        crop_gab: np.ndarray,
        crop_test: np.ndarray,
        aoi_epicenters: list | None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        height, width = crop_gab.shape[:2]
        x1, y1, x2, y2 = 0, 0, width, height

        if aoi_epicenters:
            ex, ey, ew, eh = aoi_epicenters[0]
            candidate_x1 = max(0, int(ex))
            candidate_y1 = max(0, int(ey))
            candidate_x2 = min(width, int(ex + ew))
            candidate_y2 = min(height, int(ey + eh))
            if candidate_x2 > candidate_x1 and candidate_y2 > candidate_y1:
                x1, y1, x2, y2 = (
                    candidate_x1,
                    candidate_y1,
                    candidate_x2,
                    candidate_y2,
                )

        focus_gab = crop_gab[y1:y2, x1:x2].copy()
        focus_test = crop_test[y1:y2, x1:x2].copy()
        if focus_gab.size < 50 or focus_test.size < 50:
            focus_gab = crop_gab.copy()
            focus_test = crop_test.copy()
            x1, y1, x2, y2 = 0, 0, width, height

        focus_gab, focus_test = SemanticExpert._safe_pair(focus_gab, focus_test)
        return focus_gab, focus_test, (x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def _generate_pseudo_embedding(img: np.ndarray) -> np.ndarray:
        """Converte uma imagem no embedding 128D usado pelo projeto."""
        img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        blocks_edges = []
        blocks_gray = []
        grid_size = 16
        for y in range(0, 64, grid_size):
            for x in range(0, 64, grid_size):
                block_edges = edges[y : y + grid_size, x : x + grid_size]
                block_gray = gray[y : y + grid_size, x : x + grid_size]
                blocks_edges.append(float(np.mean(block_edges) / 255.0))
                blocks_gray.append(float(np.mean(block_gray) / 255.0))

        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        histograms = []
        for channel, limit in ((0, 180), (1, 256), (2, 256)):
            histogram = cv2.calcHist([hsv], [channel], None, [32], [0, limit])
            cv2.normalize(histogram, histogram)
            histograms.append(histogram.flatten())

        return np.concatenate(
            [
                np.asarray(blocks_edges, dtype=np.float32),
                np.asarray(blocks_gray, dtype=np.float32),
                *histograms,
            ]
        ).astype(np.float32)

    @staticmethod
    def _relative_delta(
        reference: np.ndarray,
        query: np.ndarray,
        floor: float = 0.04,
    ) -> np.ndarray:
        absolute = np.abs(query - reference)
        scale = np.abs(query) + np.abs(reference) + floor
        return np.clip(absolute / scale, 0.0, 1.0)

    @classmethod
    def _dimension_descriptor(cls, index: int) -> dict:
        if index < 16:
            row, column = divmod(index, 4)
            return {
                "group": "edge_density",
                "label": f"EDGE[R{row}C{column}]",
                "local_index": index,
                "row": row,
                "column": column,
            }
        if index < 32:
            local = index - 16
            row, column = divmod(local, 4)
            return {
                "group": "brightness",
                "label": f"LUMA[R{row}C{column}]",
                "local_index": local,
                "row": row,
                "column": column,
            }

        group_specs = (
            (32, 64, "hue_histogram", "HUE"),
            (64, 96, "saturation_histogram", "SAT"),
            (96, 128, "value_histogram", "VAL"),
        )
        for start, end, group, prefix in group_specs:
            if start <= index < end:
                local = index - start
                return {
                    "group": group,
                    "label": f"{prefix}[{local:02d}]",
                    "local_index": local,
                }
        return {"group": "unknown", "label": f"DIM[{index}]", "local_index": index}

    @classmethod
    def _group_metrics(
        cls,
        reference: np.ndarray,
        query: np.ndarray,
    ) -> dict:
        metrics = {}
        for name, start, end, short_label in cls.GROUPS:
            ref_group = reference[start:end]
            query_group = query[start:end]
            delta = np.abs(query_group - ref_group)
            denominator = float(
                np.linalg.norm(ref_group) + np.linalg.norm(query_group) + 1e-8
            )
            relative = float(
                np.clip((2.0 * np.linalg.norm(delta)) / denominator, 0.0, 1.0)
            )
            metrics[name] = {
                "label": short_label,
                "start": start,
                "end": end,
                "dimensions": end - start,
                "mean_abs_delta": float(np.mean(delta)),
                "max_abs_delta": float(np.max(delta)),
                "relative_divergence": relative,
            }
        return metrics

    @staticmethod
    def _spatial_reconstruction(
        reference: np.ndarray,
        query: np.ndarray,
        output_shape: tuple[int, int],
    ) -> tuple[np.ndarray, dict]:
        edge_delta = SemanticExpert._relative_delta(reference[:16], query[:16], 0.08)
        brightness_delta = SemanticExpert._relative_delta(
            reference[16:32],
            query[16:32],
            0.10,
        )
        edge_grid = edge_delta.reshape(4, 4)
        brightness_grid = brightness_delta.reshape(4, 4)
        combined_grid = np.clip(
            edge_grid * 0.62 + brightness_grid * 0.38,
            0.0,
            1.0,
        )

        peak_flat = int(np.argmax(combined_grid))
        peak_row, peak_column = divmod(peak_flat, 4)
        peak_value = float(combined_grid[peak_row, peak_column])

        weights = combined_grid.astype(np.float64)
        total = float(weights.sum())
        centroid = None
        if total > 1e-9:
            rows, columns = np.indices((4, 4))
            centroid = [
                float((columns * weights).sum() / total),
                float((rows * weights).sum() / total),
            ]

        threshold = max(0.18, peak_value * 0.55)
        active = combined_grid >= threshold
        approximate_box = None
        if np.any(active) and peak_value > 0.04:
            active_rows, active_columns = np.where(active)
            height, width = output_shape
            x1 = int(np.floor(active_columns.min() * width / 4.0))
            y1 = int(np.floor(active_rows.min() * height / 4.0))
            x2 = int(np.ceil((active_columns.max() + 1) * width / 4.0))
            y2 = int(np.ceil((active_rows.max() + 1) * height / 4.0))
            approximate_box = [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]

        height, width = output_shape
        reconstruction = cv2.resize(
            combined_grid.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        reconstruction_map = np.clip(reconstruction * 255.0, 0, 255).astype(np.uint8)

        cells = []
        for row in range(4):
            for column in range(4):
                cells.append(
                    {
                        "row": row,
                        "column": column,
                        "edge_delta": float(edge_grid[row, column]),
                        "brightness_delta": float(brightness_grid[row, column]),
                        "combined_delta": float(combined_grid[row, column]),
                    }
                )
        cells.sort(key=lambda item: item["combined_delta"], reverse=True)

        return reconstruction_map, {
            "grid_shape": [4, 4],
            "edge_delta_grid": edge_grid.tolist(),
            "brightness_delta_grid": brightness_grid.tolist(),
            "combined_delta_grid": combined_grid.tolist(),
            "peak_cell": {
                "row": peak_row,
                "column": peak_column,
                "value": peak_value,
            },
            "centroid_grid": centroid,
            "approximate_box": approximate_box,
            "top_cells": cells[:6],
        }

    @staticmethod
    def _build_reconstruction_view(
        focus_test: np.ndarray,
        reconstruction_map: np.ndarray,
        approximate_box: list | None,
    ) -> np.ndarray:
        heatmap = cv2.applyColorMap(reconstruction_map, cv2.COLORMAP_TURBO)
        strength = reconstruction_map.astype(np.float32) / 255.0
        alpha = np.power(strength, 0.75) * 0.86
        alpha[reconstruction_map < 18] = 0.0
        alpha = alpha[:, :, None]

        output = (
            focus_test.astype(np.float32) * (1.0 - alpha)
            + heatmap.astype(np.float32) * alpha
        )
        output = np.clip(output, 0, 255).astype(np.uint8)

        height, width = focus_test.shape[:2]
        for step in range(1, 4):
            x = int(round(step * width / 4.0))
            y = int(round(step * height / 4.0))
            cv2.line(output, (x, 0), (x, height - 1), (90, 90, 90), 1)
            cv2.line(output, (0, y), (width - 1, y), (90, 90, 90), 1)

        if approximate_box:
            x, y, box_width, box_height = approximate_box
            cv2.rectangle(
                output,
                (x, y),
                (min(width - 1, x + box_width), min(height - 1, y + box_height)),
                (0, 220, 255),
                1,
                lineType=cv2.LINE_AA,
            )
        return output

    @classmethod
    def _build_debug_payload(
        cls,
        reference: np.ndarray,
        query: np.ndarray,
        semantic_distance: float,
        semantic_loss: float,
        focus_shape: tuple[int, int],
        focus_box: tuple[int, int, int, int],
        category: str,
    ) -> tuple[dict, np.ndarray]:
        absolute_delta = np.abs(query - reference)
        relative_delta = cls._relative_delta(reference, query)
        groups = cls._group_metrics(reference, query)
        dominant_group = max(
            groups,
            key=lambda name: groups[name]["relative_divergence"],
        )

        height, width = focus_shape
        reconstruction_map, spatial = cls._spatial_reconstruction(
            reference,
            query,
            (height, width),
        )

        ranked_indices = np.argsort(absolute_delta)[::-1][:12]
        top_dimensions = []
        for index in ranked_indices:
            descriptor = cls._dimension_descriptor(int(index))
            top_dimensions.append(
                {
                    **descriptor,
                    "index": int(index),
                    "reference": float(reference[index]),
                    "query": float(query[index]),
                    "signed_delta": float(query[index] - reference[index]),
                    "absolute_delta": float(absolute_delta[index]),
                    "relative_delta": float(relative_delta[index]),
                }
            )

        debug = {
            "schema": cls.SCHEMA_VERSION,
            "embedding_size": cls.EMBEDDING_SIZE,
            "category": str(category or "Unknown"),
            "focus_box": [int(value) for value in focus_box],
            "focus_size": [int(width), int(height)],
            "distance_cosine": float(semantic_distance),
            "semantic_loss": float(semantic_loss),
            "dominant_group": dominant_group,
            "groups": groups,
            "feature_ranges": {
                name: {
                    "start": start,
                    "end": end,
                    "dimensions": end - start,
                    "label": label,
                }
                for name, start, end, label in cls.GROUPS
            },
            "delta_vector": relative_delta.tolist(),
            "absolute_delta_vector": absolute_delta.tolist(),
            "histogram_delta": {
                "hue": absolute_delta[32:64].tolist(),
                "saturation": absolute_delta[64:96].tolist(),
                "value": absolute_delta[96:128].tolist(),
            },
            "spatial": spatial,
            "top_dimensions": top_dimensions,
        }
        return debug, reconstruction_map

    def analyze(
        self,
        crop_gab: np.ndarray,
        crop_test: np.ndarray,
        global_box_info: dict | None = None,
        aoi_info: dict | None = None,
        aoi_epicenters: list | None = None,
    ) -> dict:
        default_return = {
            "is_defect": False,
            "score": 0.0,
            "reason": "Imagem nula",
            "bounding_box": None,
            "semantic_loss": 0.0,
            "query_emb": None,
            "ref_emb": None,
            "semantic_debug": None,
        }

        try:
            if (
                crop_gab is None
                or crop_test is None
                or crop_gab.size == 0
                or crop_test.size == 0
            ):
                return default_return

            crop_gab, crop_test = self._safe_pair(crop_gab, crop_test)
            focus_gab, focus_test, focus_box = self._extract_focus(
                crop_gab,
                crop_test,
                aoi_epicenters,
            )

            reference_embedding = self._generate_pseudo_embedding(focus_gab)
            query_embedding = self._generate_pseudo_embedding(focus_test)

            semantic_distance = float(cosine(reference_embedding, query_embedding))
            if np.isnan(semantic_distance):
                semantic_distance = 0.0
            semantic_loss = float(min(1.0, semantic_distance * 2.5))
            is_defect = semantic_loss > 0.45

            category = ""
            if aoi_info:
                category = str(aoi_info.get("category", ""))

            semantic_debug, reconstruction_map = self._build_debug_payload(
                reference_embedding,
                query_embedding,
                semantic_distance,
                semantic_loss,
                focus_test.shape[:2],
                focus_box,
                category,
            )
            reconstruction_view = self._build_reconstruction_view(
                focus_test,
                reconstruction_map,
                semantic_debug["spatial"]["approximate_box"],
            )

            return {
                "is_defect": is_defect,
                "score": semantic_loss,
                "reason": f"Distância Semântica: {semantic_loss:.0%}",
                "bounding_box": None,
                "semantic_loss": semantic_loss,
                "semantic_distance_cosine": semantic_distance,
                "ref_emb": reference_embedding.tolist(),
                "query_emb": query_embedding.tolist(),
                "semantic_delta": semantic_debug["delta_vector"],
                "semantic_debug": semantic_debug,
                "semantic_reconstruction_map": reconstruction_map,
                "semantic_reconstruction_view": reconstruction_view,
                "semantic_focus_test": focus_test,
                "semantic_focus_box": focus_box,
            }
        except Exception as exc:
            print(f"⚠️ Erro no SemanticExpert (Embedding Generator): {exc}")
            default_return["reason"] = f"Erro interno: {exc}"
            return default_return
