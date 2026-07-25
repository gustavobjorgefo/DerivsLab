"""Instrument registry for DerivsLab.

Maintains an in-memory mapping from instrument identifier to
``InstrumentContract``, serving as the single source of truth for
reference data within a running process.

Responsibilities
----------------
- Store and retrieve ``InstrumentContract`` instances by ``instrument_id``.
- Support bulk loading from external sources (B3 files, databases) through
  dedicated loader methods, which live in separate modules and call
  ``add()`` here.

What does NOT belong here
--------------------------
- Parsing logic for B3 CSV files — that lives in the instrument loader.
- Market data (prices, vol) — that lives in ``MarketSnapshot``.
- Persistence — the registry is rebuilt on each process start from the
  authoritative source (S3, local cache, etc.).
"""

from __future__ import annotations

from typing import Iterator

from derivslab.instruments.contracts import InstrumentContract


class InstrumentRegistry:
    """In-memory registry of ``InstrumentContract`` instances.

    Keyed by ``instrument_id`` (the B3 ticker or internal identifier
    recorded on the contract). Duplicate additions overwrite the existing
    entry — last write wins, which simplifies incremental reloads from
    updated B3 files without requiring explicit deletes.

    Parameters
    ----------
    None — start empty; populate via ``add`` or ``load``.

    Examples
    --------
    >>> registry = InstrumentRegistry()
    >>> registry.add(equity_contract)
    >>> contract = registry.get("PETR4")
    """

    def __init__(self) -> None:
        self._contracts: dict[str, InstrumentContract] = {}

    # --- write API ----------------------------------------------------------

    def add(self, contract: InstrumentContract) -> None:
        """Register a single contract.

        If a contract with the same ``instrument_id`` already exists, it is
        silently replaced.

        Parameters
        ----------
        contract : InstrumentContract
            The contract to register.
        """
        self._contracts[contract.instrument_id] = contract

    def load(self, contracts: list[InstrumentContract]) -> None:
        """Register a collection of contracts in a single call.

        Convenience wrapper around repeated ``add`` calls. Duplicate
        ``instrument_id`` values within ``contracts`` are resolved in
        iteration order — the last one wins.

        Parameters
        ----------
        contracts : list[InstrumentContract]
            Contracts to register.
        """
        for contract in contracts:
            self.add(contract)

    # --- read API -----------------------------------------------------------

    def get(self, symbol: str) -> InstrumentContract | None:
        """Return the contract for *symbol*, or ``None`` if not found.

        Parameters
        ----------
        symbol : str
            Instrument identifier — must match ``InstrumentContract.instrument_id``
            exactly (case-sensitive).

        Returns
        -------
        InstrumentContract or None
        """
        return self._contracts.get(symbol)

    def all_contracts(self) -> list[InstrumentContract]:
        """Return all registered contracts.

        Returns
        -------
        list[InstrumentContract]
            Snapshot of the registry contents at the time of the call.
        """
        return list(self._contracts.values())

    # --- dunder -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._contracts)

    def __contains__(self, symbol: object) -> bool:
        return symbol in self._contracts

    def __iter__(self) -> Iterator[str]:
        return iter(self._contracts)

    def __repr__(self) -> str:
        return f"InstrumentRegistry({len(self)} instruments)"