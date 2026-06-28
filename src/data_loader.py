"""
data_loader.py
==============
Comprehensive data loading and extraction module for Tanzania BEST datasets.
Handles the heterogeneous Excel workbook formats across 2020-2024, extracts
all relevant secondary education indicators, and returns harmonised DataFrames.

Author : Habil Masawika
Project: Forecasting Education Performance Using Tanzania BEST Data (2020-2024)
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_loader")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_REGIONS: List[str] = [
    "ARUSHA", "DAR ES SALAAM", "DODOMA", "GEITA", "IRINGA", "KAGERA",
    "KATAVI", "KIGOMA", "KILIMANJARO", "LINDI", "MANYARA", "MARA",
    "MBEYA", "MOROGORO", "MTWARA", "MWANZA", "NJOMBE", "PWANI",
    "RUKWA", "RUVUMA", "SHINYANGA", "SIMIYU", "SINGIDA", "SONGWE",
    "TABORA", "TANGA",
]

REGION_ALIASES: Dict[str, str] = {
    "DAR ES SALAAM": "DAR ES SALAAM",
    "DARES SALAAM":  "DAR ES SALAAM",
    "DAR-ES-SALAAM": "DAR ES SALAAM",
    "DAR_ES_SALAAM": "DAR ES SALAAM",
}

# Sheet name mappings across years (canonical_name -> {year: actual_sheet_name})
SHEET_MAP: Dict[str, Dict[int, str]] = {
    "schools_region": {
        2020: "Table 74",
        2021: "T3.1#ofschools",
        2022: "T3.1#ofschools",
        2023: "T3.1#ofschools",
        2024: "T3.1#ofschools",
    },
    "enrolment_total": {
        2020: "Table 80",
        2021: "T3.2EnrolGTot",
        2022: "T3.2EnrolTot",
        2023: "T3.2EnrolTot",
        2024: "T3.2EnrolTot1",
    },
    "csee_pass_rate": {
        2020: "Table 99",
        2021: "T3.25PassRateCSEE",
        2022: "T3.25PassRateCSEE",
        2023: "T3.25PassRateCSEE",
        2024: "T3.27PassRateCSEE",
    },
    "teacher_ptr": {
        2020: "Table 105",
        2021: "T3.30Tot-QPTR&PTR",
        2022: "T3.30Tot-QPTR&PTR",
        2023: "T3.30Tot-QPTR&PTR",
        2024: "T3.32Tot-QPTR&PTR",
    },
    "teacher_region": {
        2020: "Table 106",
        2021: "T3.31TotTeachAge",
        2022: "T3.31TotTeachAge",
        2023: "T3.31TotTeachAge",
        2024: "T3.33TotTeachAge",
    },
    "electricity_region": {
        2021: "T3.39Tot-Electric",
        2022: "T3.39Tot-Electric",
        2023: "T3.39Tot-Electric",
        2024: "T3.41Tot-Electric ",
    },
    "ict_region": {
        2021: "T3.41Tot-ICTEquip",
        2022: "T3.41Tot-ICTEquip",
        2023: "T3.41Tot-ICTEquip",
        2024: "T3.43Tot-ICTEquip",
    },
    "completion_rate": {
        2021: "T3.7ComplRateF4",
        2022: "T3.7ComplRateF4",
        2023: "T3.7ComplRateF4",
        2024: "T3.7ComplRateF4",
    },
    "dropout_region": {
        2020: "Table 108",
        2021: "T3.19TotDropReg",
        2022: "T3.19TotDropReg",
        2023: "T3.19TotDropReg",
        2024: "T3.21TotDropReg",
    },
    "dropout_national": {
        2021: "T3.18Drop&Cht3.6",
        2022: "T3.18Drop&Cht3.6",
        2023: "T3.18Drop&Cht3.6",
        2024: "T3.20Drop&Cht3.6",
    },
    "textbook_ratio": {
        2021: "T3.47Tot-PBR",
        2022: "T3.47Tot-PBR",
        2023: "T3.47Tot-PBR",
        2024: "T3.49Tot-PBR",
    },
    "csee_by_subject": {
        2021: "T3.27PR CSEE",
        2022: "T3.27PR CSEE",
        2023: "T3.27PR CSEE",
        2024: "T3.29PR CSEE",
    },
    "disability_region": {
        2021: "T3.10TotDisabReg",
        2022: "T3.10TotDisabReg",
        2023: "T3.10TotDisabReg",
        2024: "T3.12TotDisabReg",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_region(name: object) -> Optional[str]:
    """Normalise a region name to canonical upper-case form."""
    if not isinstance(name, str):
        return None
    name = name.strip().upper()
    return REGION_ALIASES.get(name, name)


def is_region_row(val: object) -> bool:
    """Return True if val matches a known Tanzania region name."""
    return clean_region(val) in EXPECTED_REGIONS


def to_num(val: object) -> float:
    """Coerce val to float; return NaN on failure."""
    return pd.to_numeric(val, errors="coerce")


def safe_read(xl: pd.ExcelFile, sheet: str, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Read a sheet; return None and log if sheet not found."""
    try:
        return pd.read_excel(xl, sheet_name=sheet, header=None, nrows=nrows)
    except Exception as e:
        logger.debug(f"Could not read sheet '{sheet}': {e}")
        return None


def get_sheet(xl: pd.ExcelFile, year: int, canonical: str) -> Optional[pd.DataFrame]:
    """Look up the correct sheet name for a given year and read it."""
    mapping = SHEET_MAP.get(canonical, {})
    sheet = mapping.get(year)
    if sheet is None:
        logger.debug(f"No sheet mapping for {canonical} / {year}")
        return None
    # Handle trailing spaces in 2024 sheet names
    candidates = [sheet, sheet.strip()]
    for c in candidates:
        if c in xl.sheet_names:
            return safe_read(xl, c)
    # Fuzzy match
    for s in xl.sheet_names:
        if s.strip() == sheet.strip():
            return safe_read(xl, s)
    logger.debug(f"Sheet '{sheet}' not found for year {year}")
    return None


# ---------------------------------------------------------------------------
# Workbook loading
# ---------------------------------------------------------------------------
class BESTLoader:
    """
    Loads and extracts all relevant indicators from Tanzania BEST Excel files.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the five BEST Excel files.
    years : list of int
        Years to load (default: [2020, 2021, 2022, 2023, 2024]).
    """

    FILE_MAP = {
        2020: "BEST_2020_National_Data.xlsx",
        2021: "BEST_2021_National_Data.xlsx",
        2022: "BEST_2022_National_Data.xlsx",
        2023: "BEST_2023_National_Data.xlsx",
        2024: "BEST_2024_National_Data.xlsx",
    }

    def __init__(self, data_dir: str = "/mnt/user-data/uploads",
                 years: Optional[List[int]] = None):
        self.data_dir = Path(data_dir)
        self.years = years or [2020, 2021, 2022, 2023, 2024]
        self.workbooks: Dict[int, pd.ExcelFile] = {}
        self._load_workbooks()

    def _load_workbooks(self) -> None:
        """Open all Excel files and store ExcelFile handles."""
        for yr in self.years:
            path = self.data_dir / self.FILE_MAP[yr]
            if path.exists():
                self.workbooks[yr] = pd.ExcelFile(path)
                logger.info(f"Loaded {yr}: {len(self.workbooks[yr].sheet_names)} sheets")
            else:
                logger.warning(f"File not found: {path}")

    def sheet_inventory(self) -> pd.DataFrame:
        """Return a DataFrame summarising sheet presence across years."""
        all_sheets = set()
        for xl in self.workbooks.values():
            all_sheets.update(s.strip() for s in xl.sheet_names)
        rows = []
        for sheet in sorted(all_sheets):
            row = {"sheet": sheet}
            for yr in self.years:
                xl = self.workbooks.get(yr)
                if xl:
                    present = any(s.strip() == sheet for s in xl.sheet_names)
                    row[str(yr)] = "Y" if present else "-"
                else:
                    row[str(yr)] = "?"
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Extractor: Schools by region
    # ------------------------------------------------------------------
    def extract_schools_region(self, year: int) -> pd.DataFrame:
        """Extract school counts and enrolment by region."""
        xl = self.workbooks.get(year)
        if xl is None:
            return pd.DataFrame()
        df = get_sheet(xl, year, "schools_region")
        if df is None:
            return pd.DataFrame()
        records = []
        col_offsets = (1, 2, 3, 4) if year == 2020 else (2, 3, 4, 5)
        for _, row in df.iterrows():
            if not is_region_row(row.iloc[0]):
                continue
            reg = clean_region(row.iloc[0])
            try:
                govt = to_num(row.iloc[col_offsets[0]])
                nong = to_num(row.iloc[col_offsets[1]])
                totl = to_num(row.iloc[col_offsets[2]])
                enrl = to_num(row.iloc[col_offsets[3]])
            except IndexError:
                continue
            records.append({
                "year": year, "region": reg,
                "govt_schools": govt, "nongovt_schools": nong,
                "total_schools": totl, "enrolment_f1f4": enrl,
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Extractor: National enrolment breakdown (M/F totals by grade)
    # ------------------------------------------------------------------
    def extract_enrolment_national(self, year: int) -> Dict[str, float]:
        """Extract Form 1-6 total enrolment M/F/T from the national table."""
        xl = self.workbooks.get(year)
        if xl is None:
            return {}
        df = get_sheet(xl, year, "enrolment_total")
        if df is None:
            return {}
        result: Dict[str, float] = {"year": year}
        for _, row in df.iterrows():
            label = str(row.iloc[0]).strip().lower()
            if "grand total" in label or "a) form 1-6" in label or "form 1-6" in label:
                if "grand total" in label:
                    nums = [to_num(v) for v in row[1:] if pd.notna(to_num(v))]
                    if len(nums) >= 3:
                        # Last year's M, F, T cluster
                        result["enrol_male_total"]   = nums[-3]
                        result["enrol_female_total"] = nums[-2]
                        result["enrol_total"]        = nums[-1]
                    break
        return result

    # ------------------------------------------------------------------
    # Extractor: CSEE national pass rate trend
    # ------------------------------------------------------------------
    def extract_csee_national(self) -> pd.DataFrame:
        """Extract the national CSEE pass rate time series from all years."""
        records = []
        for year in self.years:
            xl = self.workbooks.get(year)
            if xl is None:
                continue
            df = get_sheet(xl, year, "csee_pass_rate")
            if df is None:
                continue
            for _, row in df.iterrows():
                yr_val = to_num(row.iloc[0])
                if pd.isna(yr_val) or not (2010 <= yr_val <= 2025):
                    continue
                # Column 6 = Percent Passed (values like 85.84, 87.30, 89.37)
                # Column 7 = Percent Failed
                # Column 8 = Candidates examined
                pct_passed = to_num(row.iloc[6]) if len(row) > 6 else np.nan
                pct_failed = to_num(row.iloc[7]) if len(row) > 7 else np.nan
                candidates = to_num(row.iloc[8]) if len(row) > 8 else np.nan
                div1       = to_num(row.iloc[1]) if len(row) > 1 else np.nan

                # Validate: pass rate should be between 40 and 100
                if pd.notna(pct_passed) and 40 <= pct_passed <= 100:
                    records.append({
                        "year":             int(yr_val),
                        "csee_pass_rate":   pct_passed,
                        "csee_fail_rate":   pct_failed if pd.notna(pct_failed) else 100 - pct_passed,
                        "csee_div1_pct":    div1 if pd.notna(div1) and 0 < div1 < 20 else np.nan,
                        "csee_candidates":  candidates,
                    })

        df_out = (pd.DataFrame(records)
                    .drop_duplicates(subset="year", keep="last")
                    .sort_values("year")
                    .reset_index(drop=True))
        logger.info(f"CSEE national: {len(df_out)} year-rows extracted")
        return df_out

    # ------------------------------------------------------------------
    # Extractor: Teachers by region
    # ------------------------------------------------------------------
    def extract_teachers_region(self, year: int) -> pd.DataFrame:
        """Extract teacher count totals by region."""
        xl = self.workbooks.get(year)
        if xl is None:
            return pd.DataFrame()
        df = get_sheet(xl, year, "teacher_region")
        if df is None:
            return pd.DataFrame()
        records = []
        for _, row in df.iterrows():
            if not is_region_row(row.iloc[0]):
                continue
            reg = clean_region(row.iloc[0])
            nums = [(i, to_num(v)) for i, v in enumerate(row) if i > 0]
            valid = [(i, v) for i, v in nums if pd.notna(v)]
            if valid:
                total_teachers = valid[-1][1]
                records.append({"year": year, "region": reg,
                                "total_teachers": total_teachers})
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Extractor: National teacher summary (PTR, qualified)
    # ------------------------------------------------------------------
    def extract_teacher_summary(self, year: int) -> Dict[str, float]:
        """Extract national PTR, total teachers, and qualified teachers."""
        xl = self.workbooks.get(year)
        if xl is None:
            return {"year": year}
        df = get_sheet(xl, year, "teacher_ptr")
        if df is None:
            return {"year": year}
        result: Dict[str, object] = {"year": year}

        in_qualified_section = False

        for _, row in df.iterrows():
            label = str(row.iloc[0]).strip().lower()
            # All numeric values larger than 1000 (teacher counts)
            big_nums = [(i, pd.to_numeric(v, errors="coerce")) for i, v in enumerate(row)
                        if i > 0 and pd.notna(pd.to_numeric(v, errors="coerce"))
                        and pd.to_numeric(v, errors="coerce") > 1000]
            # PTR strings
            ptr_strs = [str(v) for v in row[1:] if str(v).strip().startswith("1:")]

            if "total teachers" in label:
                in_qualified_section = False
                if big_nums:
                    # The last triplet's T column = last value in groups of 3 (M,F,T)
                    # Take the largest valid total (last T of the most recent year)
                    # Groups of 3: pick last element of last complete triple
                    vals = [v for _, v in big_nums]
                    # In groups of 3, the totals (T) are at positions 2, 5, 8, ...
                    totals = [vals[i] for i in range(2, len(vals), 3) if i < len(vals)]
                    if totals:
                        result["total_teachers_national"] = totals[-1]
                    else:
                        result["total_teachers_national"] = big_nums[-1][1]

            elif "qualified teachers" in label:
                in_qualified_section = True

            elif in_qualified_section and label in ["total", "grand total"]:
                if big_nums:
                    vals = [v for _, v in big_nums]
                    totals = [vals[i] for i in range(2, len(vals), 3) if i < len(vals)]
                    if totals:
                        result["qualified_teachers_national"] = totals[-1]
                    else:
                        result["qualified_teachers_national"] = big_nums[-1][1]
                in_qualified_section = False

            elif "pupil teacher ratio" in label or "ptr" in label.replace(" ", ""):
                if ptr_strs:
                    try:
                        result["ptr_national"] = float(
                            ptr_strs[-1].replace("1:", "").strip()
                        )
                    except ValueError:
                        pass

        return result

    # ------------------------------------------------------------------
    # Extractor: Electricity access by region
    # ------------------------------------------------------------------
    def extract_electricity_region(self, year: int) -> pd.DataFrame:
        """Extract % of schools with electricity by region."""
        xl = self.workbooks.get(year)
        if xl is None:
            return pd.DataFrame()
        df = get_sheet(xl, year, "electricity_region")
        if df is None:
            return pd.DataFrame()
        records = []
        for _, row in df.iterrows():
            if not is_region_row(row.iloc[0]):
                continue
            reg = clean_region(row.iloc[0])
            # Last column with value in (0, 100] is the percentage
            cands = [(i, to_num(v)) for i, v in enumerate(row)
                     if i > 1 and 0 < to_num(v) <= 100 and pd.notna(to_num(v))]
            n_schools = to_num(row.iloc[1])
            if cands:
                records.append({
                    "year": year, "region": reg,
                    "schools_with_electricity": n_schools,
                    "pct_schools_electricity": cands[-1][1],
                })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Extractor: ICT equipment by region
    # ------------------------------------------------------------------
    def extract_ict_region(self, year: int) -> pd.DataFrame:
        """Extract ICT equipment counts by region."""
        xl = self.workbooks.get(year)
        if xl is None:
            return pd.DataFrame()
        df = get_sheet(xl, year, "ict_region")
        if df is None:
            return pd.DataFrame()
        records = []
        for _, row in df.iterrows():
            if not is_region_row(row.iloc[0]):
                continue
            reg = clean_region(row.iloc[0])
            records.append({
                "year": year, "region": reg,
                "desktop_computers": to_num(row.iloc[2]) if len(row) > 2 else np.nan,
                "laptop_computers":  to_num(row.iloc[3]) if len(row) > 3 else np.nan,
                "projectors":        to_num(row.iloc[4]) if len(row) > 4 else np.nan,
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Extractor: Dropout by region
    # ------------------------------------------------------------------
    def extract_dropout_region(self, year: int) -> pd.DataFrame:
        """Extract dropout counts and rates by region."""
        xl = self.workbooks.get(year)
        if xl is None:
            return pd.DataFrame()
        df = get_sheet(xl, year, "dropout_region")
        if df is None:
            return pd.DataFrame()
        records = []
        enrolment_col_present = False
        for _, row in df.iterrows():
            if not is_region_row(row.iloc[0]):
                continue
            reg = clean_region(row.iloc[0])
            # Col 1 = enrolment, subsequent cols = dropout breakdown
            enrolment = to_num(row.iloc[1])
            nums = [to_num(v) for v in row[2:]]
            valid = [v for v in nums if pd.notna(v) and v >= 0]
            total_dropouts = valid[-1] if valid else np.nan
            records.append({
                "year": year, "region": reg,
                "enrolment_for_dropout": enrolment,
                "total_dropouts": total_dropouts,
            })
        df_out = pd.DataFrame(records)
        if not df_out.empty and "enrolment_for_dropout" in df_out.columns:
            df_out["dropout_rate_regional"] = (
                df_out["total_dropouts"] / df_out["enrolment_for_dropout"] * 100
            ).round(4)
        return df_out

    # ------------------------------------------------------------------
    # Extractor: National dropout rate
    # ------------------------------------------------------------------
    def extract_dropout_national(self) -> pd.DataFrame:
        """Extract national dropout rate series."""
        # Known values from BEST notes
        known = {2020: 4.59}
        records = []
        for year in self.years:
            if year == 2020:
                records.append({"year": 2020, "dropout_rate_pct": 4.59})
                continue
            xl = self.workbooks.get(year)
            if xl is None:
                continue
            df = get_sheet(xl, year, "dropout_national")
            if df is None:
                continue
            for _, row in df.iterrows():
                if "grand total" in str(row.iloc[0]).lower():
                    cands = [(i, to_num(v)) for i, v in enumerate(row)
                             if i > 3 and 0 < to_num(v) < 15 and pd.notna(to_num(v))]
                    if cands:
                        records.append({"year": year, "dropout_rate_pct": cands[0][1]})
                        break
        return (pd.DataFrame(records)
                  .drop_duplicates(subset="year")
                  .sort_values("year")
                  .reset_index(drop=True))

    # ------------------------------------------------------------------
    # Extractor: Completion rate
    # ------------------------------------------------------------------
    def extract_completion_rate(self) -> pd.DataFrame:
        """Extract national gross completion rate series."""
        # 2020 from 2021 BEST
        records = [{"year": 2020, "gross_completion_rate": 38.55}]
        for year in [2021, 2022, 2023, 2024]:
            xl = self.workbooks.get(year)
            if xl is None:
                continue
            df = get_sheet(xl, year, "completion_rate")
            if df is None:
                continue
            for _, row in df.iterrows():
                if "gross completion" in str(row.iloc[0]).lower():
                    nums = [to_num(v) for v in row[1:]]
                    vals = [v for v in nums if pd.notna(v) and 10 < v < 100]
                    if vals:
                        records.append({"year": year, "gross_completion_rate": vals[-1]})
                        break
        return (pd.DataFrame(records)
                  .drop_duplicates(subset="year")
                  .sort_values("year")
                  .reset_index(drop=True))

    # ------------------------------------------------------------------
    # Extractor: Textbook/pupil-book ratio
    # ------------------------------------------------------------------
    def extract_textbook_ratio(self, year: int) -> pd.DataFrame:
        """Extract pupil-book ratio by region."""
        xl = self.workbooks.get(year)
        if xl is None:
            return pd.DataFrame()
        df = get_sheet(xl, year, "textbook_ratio")
        if df is None:
            return pd.DataFrame()
        records = []
        for _, row in df.iterrows():
            if not is_region_row(row.iloc[0]):
                continue
            reg = clean_region(row.iloc[0])
            nums = [to_num(v) for v in row[1:] if pd.notna(to_num(v))]
            # Last valid numeric is typically the overall PBR
            pbr = nums[-1] if nums else np.nan
            records.append({"year": year, "region": reg, "pupil_book_ratio": pbr})
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Extractor: Disability enrolment by region
    # ------------------------------------------------------------------
    def extract_disability_region(self, year: int) -> pd.DataFrame:
        """Extract disability enrolment totals by region."""
        xl = self.workbooks.get(year)
        if xl is None:
            return pd.DataFrame()
        df = get_sheet(xl, year, "disability_region")
        if df is None:
            return pd.DataFrame()
        records = []
        for _, row in df.iterrows():
            if not is_region_row(row.iloc[0]):
                continue
            reg = clean_region(row.iloc[0])
            nums = [to_num(v) for v in row[1:] if pd.notna(to_num(v))]
            total_disabled = nums[-1] if nums else np.nan
            records.append({"year": year, "region": reg,
                            "disability_enrolment": total_disabled})
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Master builder: regional panel
    # ------------------------------------------------------------------
    def build_regional_panel(self) -> pd.DataFrame:
        """
        Build the full region x year panel combining all extracted indicators.

        Returns
        -------
        pd.DataFrame
            One row per (region, year) combination with all available indicators.
        """
        logger.info("Building regional panel...")

        # Schools (all years)
        schools_frames = [self.extract_schools_region(yr) for yr in self.years]
        df_schools = pd.concat(schools_frames, ignore_index=True)
        df_schools = df_schools[df_schools["region"].isin(EXPECTED_REGIONS)]

        # Teachers by region (all years)
        teacher_frames = [self.extract_teachers_region(yr) for yr in self.years]
        df_teachers = pd.concat(teacher_frames, ignore_index=True)
        df_teachers = df_teachers[df_teachers["region"].isin(EXPECTED_REGIONS)]

        # Electricity (2021-2024, 2020 imputed)
        elec_frames = [self.extract_electricity_region(yr) for yr in [2021, 2022, 2023, 2024]]
        df_elec = pd.concat(elec_frames, ignore_index=True)
        df_elec = df_elec[df_elec["region"].isin(EXPECTED_REGIONS)]

        # ICT (2021-2024)
        ict_frames = [self.extract_ict_region(yr) for yr in [2021, 2022, 2023, 2024]]
        df_ict = pd.concat(ict_frames, ignore_index=True)
        df_ict = df_ict[df_ict["region"].isin(EXPECTED_REGIONS)]

        # Dropout by region
        drop_frames = [self.extract_dropout_region(yr) for yr in self.years]
        df_drop = pd.concat(drop_frames, ignore_index=True)
        df_drop = df_drop[df_drop["region"].isin(EXPECTED_REGIONS)]

        # Textbook ratio (2021-2024) — sheet may not exist; guard against empty
        tbk_frames = [self.extract_textbook_ratio(yr) for yr in [2021, 2022, 2023, 2024]]
        tbk_frames = [f for f in tbk_frames if not f.empty]
        if tbk_frames:
            df_tbk = pd.concat(tbk_frames, ignore_index=True)
            df_tbk = df_tbk[df_tbk["region"].isin(EXPECTED_REGIONS)]
        else:
            df_tbk = pd.DataFrame(columns=["year", "region", "pupil_book_ratio"])

        # Disability (2021-2024)
        dis_frames = [self.extract_disability_region(yr) for yr in [2021, 2022, 2023, 2024]]
        dis_frames = [f for f in dis_frames if not f.empty]
        if dis_frames:
            df_dis = pd.concat(dis_frames, ignore_index=True)
            df_dis = df_dis[df_dis["region"].isin(EXPECTED_REGIONS)]
        else:
            df_dis = pd.DataFrame(columns=["year", "region", "disability_enrolment"])

        # National time series
        df_csee       = self.extract_csee_national()
        df_dropout_n  = self.extract_dropout_national()
        df_completion = self.extract_completion_rate()
        teacher_sum   = pd.DataFrame([self.extract_teacher_summary(yr) for yr in self.years])

        national_ts = (df_csee
                       .merge(df_dropout_n,  on="year", how="outer")
                       .merge(df_completion, on="year", how="outer")
                       .merge(teacher_sum,   on="year", how="outer")
                       .sort_values("year"))

        # Merge all regional frames
        panel = (df_schools
                 .merge(df_teachers, on=["year", "region"], how="left")
                 .merge(df_elec,     on=["year", "region"], how="left")
                 .merge(df_ict,      on=["year", "region"], how="left")
                 .merge(df_drop,     on=["year", "region"], how="left")
                 .merge(df_tbk,      on=["year", "region"], how="left")
                 .merge(df_dis,      on=["year", "region"], how="left"))

        # Attach national series
        panel = panel.merge(national_ts, on="year", how="left")
        panel = panel.sort_values(["region", "year"]).reset_index(drop=True)

        logger.info(f"Regional panel built: {panel.shape[0]} rows x {panel.shape[1]} cols")
        return panel

    def save_raw_panel(self, panel: pd.DataFrame, output_dir: str = "data/raw") -> None:
        """Save the raw extracted panel to CSV."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / "best_panel_raw.csv"
        panel.to_csv(path, index=False)
        logger.info(f"Saved raw panel to {path}")


# ---------------------------------------------------------------------------
# Convenience loader (used by notebooks)
# ---------------------------------------------------------------------------
def load_best_data(data_dir: str = "/mnt/user-data/uploads") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-call convenience function: load everything and return (panel, csee_national).

    Returns
    -------
    panel : pd.DataFrame
        Region x year panel with all extracted indicators.
    csee : pd.DataFrame
        National CSEE pass rate time series.
    """
    loader = BESTLoader(data_dir=data_dir)
    panel  = loader.build_regional_panel()
    csee   = loader.extract_csee_national()
    return panel, csee
