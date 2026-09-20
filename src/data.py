"""Loading, schema validation, and cleaning of the labelled landslide dataset.

This module owns every operation that can be performed on a single row without
reference to any other row: parsing, type coercion, sentinel decoding, and the
removal of exact duplicates. Anything that must be *learned* from a set of rows
(imputation statistics, scaler parameters, category vocabularies) belongs in
``features.py``, so that it can be fit inside a training fold and never on the
full dataset.

Design notes carried from the Session 1 audit
---------------------------------------------
- The raw file is **sorted by class** (rows 0-197 are all positives). Nothing in
  this module may rely on row order, and nothing downstream may treat the index
  as meaningful. ``clean`` shuffles with a fixed seed so that a splitter which
  forgets to cannot silently produce a single-label training set.
- ``SPI`` and ``TWI`` are byte-identical in the raw file and appear to be
  z-scored rather than in physical units. ``check_spi_twi_collision`` exists to
  make that failure loud rather than silent.
- ``Aspect == -1`` is the flat-cell sentinel, not a missing value. It is decoded
  here, once, before any imputer can mistake it for data.

See ``results/session1_data_summary.md`` for the evidence behind each of these,
and ``results/session2_feature_check.md`` for the choices layered on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TARGET: str = "Landslide"
COORD_COLS: list[str] = ["Latitude", "Longitude"]
CATEGORICAL_COLS: list[str] = ["LULC", "Soil_Texture"]
CIRCULAR_COLS: list[str] = ["Aspect"]
CONTINUOUS_COLS: list[str] = [
    "Elevation",
    "Slope",
    "Curvature",
    "SPI",
    "TWI",
    "TRI",
    "NDVI",
    "Annual_Rainfall",
    "Drainage_Density",
]

ASPECT_FLAT_SENTINEL: float = -1.0
RANDOM_STATE: int = 42

#: Columns whose missingness is worth flagging. ``SPI`` disappears once
#: ``resolve_spi_twi`` has run, so the list is filtered against the frame
#: actually passed in rather than assumed present.
DEFAULT_INDICATOR_COLS: list[str] = ["SPI", "TWI", "Soil_Texture"]


@dataclass(frozen=True)
class DataSpec:
    """Declares what the cleaned frame is expected to contain.

    Holds the target name, the coordinate columns, and the categorical /
    circular / continuous groupings, so that the rest of the pipeline asks this
    object rather than hard-coding column lists in five different places.

    Attributes
    ----------
    target:
        Name of the binary label column.
    coords:
        Columns used for spatial blocking and neighbour-based imputation. These
        are deliberately *not* model features; see the audit, section 6.3.
    categorical:
        Integer class codes that must be one-hot encoded, never treated as
        ordinal.
    circular:
        Angular columns in degrees, requiring sin/cos encoding.
    continuous:
        Genuinely continuous covariates.
    """

    target: str = TARGET
    coords: list[str] = field(default_factory=lambda: list(COORD_COLS))
    categorical: list[str] = field(default_factory=lambda: list(CATEGORICAL_COLS))
    circular: list[str] = field(default_factory=lambda: list(CIRCULAR_COLS))
    continuous: list[str] = field(default_factory=lambda: list(CONTINUOUS_COLS))

    def all_columns(self) -> list[str]:
        """Every column the spec names, in a stable order."""
        return [
            *self.coords,
            *self.continuous,
            *self.circular,
            *self.categorical,
            self.target,
        ]

    def feature_columns(self, df: pd.DataFrame | None = None) -> list[str]:
        """Model-visible columns: everything but the target and the coordinates.

        Passing ``df`` filters the list to columns that actually exist, which
        matters after ``resolve_spi_twi`` has removed one of the duplicated
        topographic indices.
        """
        cols = [*self.continuous, *self.circular, *self.categorical]
        if df is not None:
            cols = [c for c in cols if c in df.columns]
        return cols


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the labelled CSV exactly as written, with no cleaning applied.

    Performs only the parsing that cannot be deferred: reading the file and
    confirming it is non-empty. Column names are left untouched so that
    ``validate_schema`` can report discrepancies against what was actually on
    disk rather than against a silently normalised version.

    Parameters
    ----------
    path:
        Path to ``Labeled Data Set.csv``.

    Returns
    -------
    The raw frame, expected to be 498 rows by 15 columns.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"labelled dataset not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"labelled dataset is empty: {path}")
    return df


def validate_schema(df: pd.DataFrame, spec: DataSpec | None = None) -> None:
    """Assert that a frame matches the expected columns, dtypes, and label domain.

    Checks that every column named in ``spec`` is present, that ``Landslide``
    contains only ``{0, 1}`` with no missing values, and that the categorical
    columns contain only integral values once NaNs are set aside. Intended to be
    called immediately after loading and again after cleaning, so that a
    transformation that quietly drops or renames a column fails here rather than
    surfacing as a confusing error inside a model fit.

    Raises
    ------
    ValueError
        With a message naming every violation found, not just the first.
    """
    spec = spec or DataSpec()
    problems: list[str] = []

    # After resolve_spi_twi only one of SPI/TWI survives, so the pair is checked
    # as "at least one present" rather than demanding both.
    required = [c for c in spec.all_columns() if c not in {"SPI", "TWI"}]
    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if not {"SPI", "TWI"} & set(df.columns):
        problems.append("neither SPI nor TWI is present")

    if spec.target in df.columns:
        tgt = df[spec.target]
        if tgt.isna().any():
            problems.append(f"{spec.target} has {int(tgt.isna().sum())} missing values")
        bad = set(pd.unique(tgt.dropna())) - {0, 1}
        if bad:
            problems.append(f"{spec.target} has values outside 0/1: {sorted(bad)}")

    for col in spec.categorical:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        # Codes arrive as float64 purely because NaNs force the dtype; the
        # values themselves must still be whole numbers.
        if len(vals) and not np.all(np.mod(vals.astype(float), 1) == 0):
            problems.append(f"{col} contains non-integral class codes")

    for col in spec.coords:
        if col in df.columns and df[col].isna().any():
            problems.append(
                f"{col} has missing values; splitting geometry needs complete coordinates"
            )

    if problems:
        raise ValueError("schema validation failed:\n  - " + "\n  - ".join(problems))


def check_spi_twi_collision(df: pd.DataFrame) -> None:
    """Fail loudly if ``SPI`` and ``TWI`` are still the same column.

    In the raw file these two are byte-identical, including their NaN masks, and
    both sit on a standardised scale that matches neither variable's physical
    units. Until the upstream raster extraction is redone, the dataset contains
    one topographic wetness/power index, not two, and any feature-importance
    result that lists both is double-counting a single input.

    This check exists so that the collision cannot be forgotten: it should be
    called from ``run.py`` at startup, and the run should refuse to proceed until
    either the columns genuinely differ or the caller has explicitly resolved the
    duplicate via ``resolve_spi_twi``.

    Raises
    ------
    ValueError
        If the two columns are identical on their shared non-null rows.
    """
    if not {"SPI", "TWI"} <= set(df.columns):
        return  # already resolved: only one of the pair survives

    if df["SPI"].equals(df["TWI"]):
        raise ValueError(
            "SPI and TWI are byte-identical (audit section 6.1): one column has "
            "overwritten the other upstream, and the survivor is z-scored rather "
            "than carried in physical units. The dataset holds 11 distinct "
            "predictors, not 12. Regenerate both from the DEM, or call "
            "resolve_spi_twi() to proceed with one honestly-labelled column."
        )


def resolve_spi_twi(df: pd.DataFrame, strategy: str = "drop_spi") -> pd.DataFrame:
    """Reduce the duplicated ``SPI``/``TWI`` pair to a single honest column.

    A stopgap for use while the DEM-derived rasters are being regenerated. No
    strategy here recovers the lost variable; each one only makes the loss
    explicit so that the modelling can continue without claiming twelve
    predictors when eleven exist.

    The default is ``"drop_spi"`` -- keep the column labelled ``TWI`` -- which is
    the Session 2 instruction. It is a naming choice, not a recovery: the
    surviving values are identical either way, and ``TWI`` is retained for
    continuity with the manuscript, not because the column has been shown to be
    a wetness index.

    Parameters
    ----------
    strategy:
        ``"drop_spi"``
            Keep ``TWI``, drop ``SPI``. The default; see above.
        ``"drop_twi"``
            The mirror image, for a sensitivity check.
        ``"rename_ambiguous"``
            Collapse both into one column named ``Topo_Index_Unresolved``, which
            is the most truthful option and the one that will read correctly in
            a results table.

    Returns
    -------
    A copy of ``df`` with the duplication removed.
    """
    out = df.copy()
    if len({"SPI", "TWI"} & set(out.columns)) < 2:
        return out

    if strategy == "drop_spi":
        return out.drop(columns=["SPI"])
    if strategy == "drop_twi":
        return out.drop(columns=["TWI"])
    if strategy == "rename_ambiguous":
        out["Topo_Index_Unresolved"] = out["TWI"]
        return out.drop(columns=["SPI", "TWI"])
    raise ValueError(
        f"unknown strategy {strategy!r}; expected 'drop_spi', 'drop_twi', "
        "or 'rename_ambiguous'"
    )


def decode_aspect_sentinel(
    df: pd.DataFrame,
    sentinel: float = ASPECT_FLAT_SENTINEL,
    mask_sentinel: bool = False,
) -> pd.DataFrame:
    """Split ``Aspect``'s flat-cell sentinel into a value and an indicator.

    The raw file encodes flat cells as ``Aspect == -1``, on exactly the 20 rows
    where ``Slope == 0``. That is a categorical statement ("this cell has no
    aspect"), not a measurement, and it must not reach an imputer, a scaler, or
    a sin/cos transform as the number -1.

    The indicator is derived from ``Slope == 0`` rather than from the sentinel
    itself, per the Session 2 instruction. Both definitions select exactly the
    same 20 rows here -- ``features.check_flat_cell_agreement`` asserts it -- but
    slope is the physical condition and the sentinel is one raster's way of
    writing it down.

    Parameters
    ----------
    mask_sentinel:
        ``False`` (default) leaves the ``-1`` values in place, carrying those
        rows as-is and letting ``features.encode_aspect`` map flat cells to
        ``(0, 0)`` on the unit circle. ``True`` replaces them with NaN, handing
        them to the imputer instead; that was the Session 1 plan, kept for a
        sensitivity check but not the default, because imputing an aspect for a
        cell that has none invents a bearing.

    Returns
    -------
    A copy of ``df`` with ``is_flat`` appended.
    """
    out = df.copy()
    out["is_flat"] = (out["Slope"] == 0).fillna(False).astype(int)

    if mask_sentinel:
        out.loc[out["Aspect"] == sentinel, "Aspect"] = np.nan
    return out


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove fully duplicated rows, keeping the first occurrence.

    The raw file contains two such pairs, at index (97, 98) and (182, 183), both
    positives and both duplicated in their coordinates as well. Left in place
    they inflate the sample size and can put the same physical point into both a
    training and a test fold.

    Returns
    -------
    The deduplicated frame and the number of rows removed, so that the count can
    be logged rather than silently absorbed.
    """
    before = len(df)
    out = df.drop_duplicates(keep="first").reset_index(drop=True)
    return out, before - len(out)


def add_missing_indicators(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Append binary ``<column>_was_missing`` flags before imputation.

    Missingness in this dataset is only weakly related to the label -- the
    largest gap is ``Soil_Texture`` at 0.393 present versus 0.440 missing -- so
    these indicators are not expected to carry much signal. They are cheap,
    though, and they turn an untestable MAR assumption into something a
    robustness check can actually measure.

    Parameters
    ----------
    columns:
        Defaults to the columns with non-trivial missingness: ``SPI``, ``TWI``,
        and ``Soil_Texture``. Names absent from ``df`` are skipped, so the
        default survives ``resolve_spi_twi`` having removed one of the pair.
    """
    out = df.copy()
    columns = columns if columns is not None else DEFAULT_INDICATOR_COLS
    for col in columns:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def shuffle_rows(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Return the frame in a deterministically shuffled order.

    The raw file is class-sorted: rows 0-197 are positives, 198-497 negatives.
    Every splitter in ``splits.py`` shuffles explicitly for that reason, and this
    function does not relieve them of it -- it is a second line of defence, so
    that a splitter refactored into relying on input order degrades to a random
    split rather than to a single-label training set.

    The index is reset, because past this point row position carries no meaning
    and an index that looked like it did would only invite misuse.
    """
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def clean(df: pd.DataFrame, spec: DataSpec | None = None) -> pd.DataFrame:
    """Apply the full row-local cleaning sequence in a fixed, auditable order.

    Runs, in order: duplicate removal, ``SPI``/``TWI`` de-duplication, ``Aspect``
    sentinel decoding, missing-indicator creation, categorical code coercion to
    a nullable integer dtype, and a fixed-seed shuffle. Deliberately does **not**
    impute, scale, or encode -- those steps learn parameters from data and so
    must happen inside a training fold.

    Returns
    -------
    A cleaned copy of ``df``, safe to hand to ``splits.py``.
    """
    spec = spec or DataSpec()

    out, _ = drop_exact_duplicates(df)
    out = resolve_spi_twi(out)  # keeps TWI, drops the duplicated SPI column
    out = decode_aspect_sentinel(out)
    out = add_missing_indicators(out)

    # Nullable Int64 rather than float64: these are class codes, and the float
    # dtype in the raw file is an artefact of NaN presence, not a claim that the
    # codes are continuous.
    for col in spec.categorical:
        if col in out.columns:
            out[col] = out[col].round().astype("Int64")

    out = shuffle_rows(out)
    validate_schema(out, spec)
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Produce the per-column audit table: dtype, missingness, cardinality.

    Regenerates the numbers reported in ``results/session1_data_summary.md`` so
    that the summary can be refreshed mechanically rather than transcribed by
    hand when the input data changes.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_missing": int(s.isna().sum()),
                "pct_missing": round(100 * s.isna().mean(), 2),
                "n_unique": int(s.nunique(dropna=True)),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("pct_missing", ascending=False).reset_index(drop=True)
