"""Fixtures for reference-data tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_anbima_xlsx(tmp_path: Path) -> Path:
    """A small synthetic spreadsheet mirroring ANBIMA's raw file shape.

    Includes valid holiday rows plus trailing footnote rows with no
    parseable date, matching the shape of the real published file
    (header + holidays + a "Fonte: ANBIMA" line + numbered notes).
    """
    rows = [
        {
            "Data": "2026-01-01",
            "Dia da Semana": "quinta-feira",
            "Feriado": "Confraternização Universal",
        },
        {"Data": "2026-04-21", "Dia da Semana": "terça-feira", "Feriado": "Tiradentes"},
        {"Data": "2026-12-25", "Dia da Semana": "sexta-feira", "Feriado": "Natal"},
        {"Data": "Fonte: ANBIMA", "Dia da Semana": None, "Feriado": None},
        {"Data": "1) Nota de rodapé qualquer.", "Dia da Semana": None, "Feriado": None},
    ]
    path = tmp_path / "sample_anbima.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return path
