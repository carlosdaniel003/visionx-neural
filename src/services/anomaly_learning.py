"""Aprendizado humano: JSON sempre, imagem apenas quando há discordância."""

from __future__ import annotations

from src.services.dataset_manager import DatasetManager


def install_anomaly_learning(control_panel_cls) -> None:
    if getattr(control_panel_cls, "_anomaly_learning_installed", False):
        return

    original_save_label = control_panel_cls.save_label

    def save_label(self, user_decision: str, source="button"):
        normalized = str(user_decision or "").strip().upper()
        if normalized not in {"OK", "NG"} or self.current_ng is None:
            return

        # Produção mantém a decisão automática, mas não cria rótulos para si mesma.
        if source == "auto":
            return original_save_label(self, normalized, source=source)

        if source == "button":
            self.send_command_to_xp("0" if normalized == "OK" else "1")

        ai_decision = (
            "NG"
            if self.current_analysis
            and self.current_analysis.get("is_defect", False)
            else "OK"
        )
        disagreement = ai_decision != normalized

        json_path = DatasetManager.save_sample(
            ng_image=self.current_ng,
            label=normalized,
            sample_image=self.current_sample,
            aoi_info=self.current_aoi_info,
            analysis=self.current_analysis,
            save_images=disagreement,
            source=source,
            ai_decision=ai_decision,
        )

        if json_path:
            self.orchestrator.reload_memory()

        self.btn_save_ok.setEnabled(False)
        self.btn_save_ng.setEnabled(False)
        self.btn_skip.setEnabled(False)
        self.is_locked = False
        self.btn_start.setText("Capturar Local (MSS)")
        self.btn_start.setEnabled(True)
        self.update_brain_status("Sistema Ocioso", False)

        suffix = "JSON + auditoria" if disagreement else "memória JSON"
        self.update_history_status(f"{normalized} ({suffix})", source)
        return json_path

    control_panel_cls.save_label = save_label
    control_panel_cls._anomaly_learning_installed = True
