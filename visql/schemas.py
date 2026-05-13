"""Database schema model + live BigQuery introspection + execution wrapper.

Implements the "Live schema introspection" design decision (slide 4):
    - Pull columns from INFORMATION_SCHEMA.COLUMNS
    - Enrich each table description with sample rows
    - Pre-execution check_tables_exist() to catch hallucinated names
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import re
import pandas as pd

# ── Public datasets ────────────────────────────────────────────────
THELOOK_DATASET = "bigquery-public-data.thelook_ecommerce"
SEC_DATASET     = "bigquery-public-data.sec_quarterly_financials"
GA_DATASET      = "bigquery-public-data.google_analytics_sample"

# ════════════════════════════════════════════════════════════════════
# DATACLASSES
# ════════════════════════════════════════════════════════════════════
@dataclass
class SchemaColumn:
    name: str
    type: str
    description: str = ""

@dataclass
class SchemaTable:
    name: str
    columns: list[SchemaColumn]
    description: str = ""
    fq_name: str = ""        # bigquery-public-data.thelook_ecommerce.users
    sample_rows: list[dict] = field(default_factory=list)

@dataclass
class DatabaseSchema:
    db_id: str               # short id used in prompts: 'thelook', 'sec', etc.
    description: str
    tables: list[SchemaTable]
    bq_project: Optional[str] = None
    bq_dataset: Optional[str] = None

    def has_variant_column(self) -> bool:
        """Used by router gate: does any table have a column suggesting A/B?"""
        variant_words = ("variant", "arm", "experiment_group", "treatment_group",
                         "ab_group", "group_assignment", "bucket", "cohort_id")
        for t in self.tables:
            for c in t.columns:
                lname = c.name.lower()
                if any(v in lname for v in variant_words):
                    return True
        return False

    def to_prompt_block(self, max_chars: int = 6000) -> str:
        """Serialize for inclusion in LLM prompt — truncated if too long."""
        parts = [f"Database: {self.db_id} — {self.description}", ""]
        for t in self.tables:
            parts.append(f"TABLE {t.name}  ({t.fq_name or t.name})")
            if t.description:
                parts.append(f"  -- {t.description}")
            for c in t.columns:
                line = f"  {c.name}: {c.type}"
                if c.description:
                    line += f"  // {c.description}"
                parts.append(line)
            if t.sample_rows:
                parts.append(f"  -- sample rows: {t.sample_rows[:2]}")
            parts.append("")
        out = "\n".join(parts)
        return out if len(out) <= max_chars else out[:max_chars] + "\n... [truncated]"

# ════════════════════════════════════════════════════════════════════
# LIVE BIGQUERY INTROSPECTION
# ════════════════════════════════════════════════════════════════════
def fetch_bq_schema(
    bq_client,
    project: str,
    dataset: str,
    db_id: str,
    description: str,
    table_filter: Optional[list[str]] = None,
    enrich_with_samples: bool = True,
    max_tables: int = 12,
) -> DatabaseSchema:
    """Fetch live schema from BigQuery's INFORMATION_SCHEMA.

    Args:
        table_filter: optional whitelist of base table names. If None,
                      pull every table in the dataset (capped at max_tables).
        enrich_with_samples: pull 2 sample rows per table to seed prompts.
    """
    fq_dataset = f"{project}.{dataset}"
    # 1) columns metadata
    sql = f"""
        SELECT table_name, column_name, data_type
        FROM `{fq_dataset}.INFORMATION_SCHEMA.COLUMNS`
        ORDER BY table_name, ordinal_position
    """
    df = bq_client.query(sql).to_dataframe()

    # 2) group into tables
    tables: list[SchemaTable] = []
    seen_tables = []
    for tname, group in df.groupby("table_name"):
        if table_filter and tname not in table_filter:
            continue
        if len(seen_tables) >= max_tables:
            break
        seen_tables.append(tname)

        cols = [SchemaColumn(name=r.column_name, type=r.data_type)
                for r in group.itertuples()]
        fq_name = f"{fq_dataset}.{tname}"

        sample_rows = []
        table_desc = ""
        if enrich_with_samples:
            try:
                sdf = bq_client.query(
                    f"SELECT * FROM `{fq_name}` LIMIT 2"
                ).to_dataframe()
                # cast to native python for cleaner JSON serialization
                sample_rows = [
                    {k: _to_python(v) for k, v in row.items()}
                    for row in sdf.to_dict(orient="records")
                ]
                table_desc = f"{len(group)} columns; live-introspected"
            except Exception as e:
                table_desc = f"sample fetch failed: {str(e)[:80]}"

        tables.append(SchemaTable(
            name=tname, columns=cols, fq_name=fq_name,
            description=table_desc, sample_rows=sample_rows,
        ))

    return DatabaseSchema(
        db_id=db_id,
        description=description,
        tables=tables,
        bq_project=project,
        bq_dataset=dataset,
    )

def _to_python(v: Any) -> Any:
    """Convert numpy / pandas types into native python for clean repr."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return str(v)
    if isinstance(v, (list, tuple)):
        return [_to_python(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_python(x) for k, x in v.items()}
    return v

# ── Convenience: load the three deck datasets at once ────────────
def default_schemas_live(bq_client) -> dict[str, DatabaseSchema]:
    """Load the three datasets featured in the slide deck."""
    print("[schemas] fetching theLook e-commerce live schema...")
    thelook = fetch_bq_schema(
        bq_client,
        project="bigquery-public-data",
        dataset="thelook_ecommerce",
        db_id="thelook",
        description="theLook e-commerce: users, products, orders, order_items, events, distribution_centers, inventory_items.",
        table_filter=["users", "products", "orders", "order_items",
                      "events", "distribution_centers", "inventory_items"],
    )

    print("[schemas] fetching SEC quarterly financials live schema...")
    sec = fetch_bq_schema(
        bq_client,
        project="bigquery-public-data",
        dataset="sec_quarterly_financials",
        db_id="sec",
        description="SEC quarterly filings (XBRL long-format). Stresses join planning across submissions/numbers/tags.",
    )

    print("[schemas] fetching Google Analytics sample live schema...")
    try:
        ga = fetch_bq_schema(
            bq_client,
            project="bigquery-public-data",
            dataset="google_analytics_sample",
            db_id="ga_sample",
            description="GA sample: nested STRUCT + repeated fields. Sharded ga_sessions_*.",
            table_filter=None,
            max_tables=4,
        )
    except Exception as e:
        print(f"[schemas]   GA fetch failed ({e}); skipping.")
        ga = None

    out = {"thelook": thelook, "sec": sec}
    if ga is not None:
        out["ga_sample"] = ga
    return out

# ════════════════════════════════════════════════════════════════════
# SYNTHESIZED A/B EXPERIMENT (slide 5)
# ════════════════════════════════════════════════════════════════════
EXPERIMENT_TABLE_NAME = "checkout_redesign_2024"

EXPERIMENT_SCHEMA = SchemaTable(
    name=EXPERIMENT_TABLE_NAME,
    columns=[
        SchemaColumn("user_id",     "INT64",     "User identifier"),
        SchemaColumn("variant",     "STRING",    "'control' or 'treatment'"),
        SchemaColumn("converted",   "INT64",     "0 or 1 — did the user convert"),
        SchemaColumn("assigned_at", "TIMESTAMP", "Variant assignment time"),
    ],
    description="Synthesized A/B test — variant assigned via FARM_FINGERPRINT mod 2; treatment has +8pp lift.",
    fq_name="",  # filled in at synth time
    sample_rows=[],
)

def thelook_with_experiment_schema(user_project: str) -> DatabaseSchema:
    """theLook + the synthesized A/B experiment table.

    Used by the router-eval to test the gate's positive case.
    """
    exp = SchemaTable(
        name=EXPERIMENT_TABLE_NAME,
        columns=EXPERIMENT_SCHEMA.columns,
        description=EXPERIMENT_SCHEMA.description,
        fq_name=f"{user_project}.visql_synth.{EXPERIMENT_TABLE_NAME}",
    )
    base = thelook_schema()
    base.tables.append(exp)
    base.db_id = "thelook_with_experiment"
    return base

def thelook_schema() -> DatabaseSchema:
    """Static fallback theLook schema — used when BQ client isn't available."""
    return DatabaseSchema(
        db_id="thelook",
        description="theLook e-commerce: orders, users, products, events.",
        tables=[
            SchemaTable("users", [
                SchemaColumn("id", "INT64"), SchemaColumn("email", "STRING"),
                SchemaColumn("age", "INT64"), SchemaColumn("gender", "STRING"),
                SchemaColumn("country", "STRING"), SchemaColumn("traffic_source", "STRING"),
                SchemaColumn("created_at", "TIMESTAMP"),
            ], fq_name=f"{THELOOK_DATASET}.users"),
            SchemaTable("products", [
                SchemaColumn("id", "INT64"), SchemaColumn("name", "STRING"),
                SchemaColumn("category", "STRING"), SchemaColumn("brand", "STRING"),
                SchemaColumn("retail_price", "FLOAT64"), SchemaColumn("cost", "FLOAT64"),
            ], fq_name=f"{THELOOK_DATASET}.products"),
            SchemaTable("orders", [
                SchemaColumn("order_id", "INT64"), SchemaColumn("user_id", "INT64"),
                SchemaColumn("status", "STRING"), SchemaColumn("created_at", "TIMESTAMP"),
                SchemaColumn("returned_at", "TIMESTAMP"),
                SchemaColumn("num_of_item", "INT64"),
            ], fq_name=f"{THELOOK_DATASET}.orders"),
            SchemaTable("order_items", [
                SchemaColumn("id", "INT64"), SchemaColumn("order_id", "INT64"),
                SchemaColumn("product_id", "INT64"), SchemaColumn("user_id", "INT64"),
                SchemaColumn("sale_price", "FLOAT64"),
                SchemaColumn("status", "STRING"), SchemaColumn("created_at", "TIMESTAMP"),
            ], fq_name=f"{THELOOK_DATASET}.order_items"),
            SchemaTable("events", [
                SchemaColumn("id", "INT64"), SchemaColumn("user_id", "INT64"),
                SchemaColumn("event_type", "STRING"), SchemaColumn("created_at", "TIMESTAMP"),
            ], fq_name=f"{THELOOK_DATASET}.events"),
        ],
        bq_project="bigquery-public-data",
        bq_dataset="thelook_ecommerce",
    )

def create_synthetic_experiment_table(bq_client, user_project: str, lift: float = 0.08) -> str:
    """Materialize the synthesized A/B test in `{user_project}.visql_synth.checkout_redesign_2024`.

    Variant assignment: FARM_FINGERPRINT(user_id) mod 2.
    Conversion: control 0.10 base rate, treatment 0.10 + lift.
    """
    fq = f"`{user_project}.visql_synth.{EXPERIMENT_TABLE_NAME}`"
    bq_client.query(f"CREATE SCHEMA IF NOT EXISTS `{user_project}.visql_synth`").result()
    sql = f"""
    CREATE OR REPLACE TABLE {fq} AS
    WITH assigned AS (
      SELECT
        u.id AS user_id,
        IF(MOD(ABS(FARM_FINGERPRINT(CAST(u.id AS STRING))), 2) = 0, 'control', 'treatment') AS variant,
        TIMESTAMP_ADD(TIMESTAMP('2024-01-01'),
                      INTERVAL CAST(MOD(ABS(FARM_FINGERPRINT(CAST(u.id AS STRING))), 60) AS INT64) DAY)
          AS assigned_at
      FROM `{THELOOK_DATASET}.users` u
      WHERE u.id IS NOT NULL
    )
    SELECT
      user_id,
      variant,
      CASE
        WHEN variant = 'treatment'
             AND RAND() < (0.10 + {lift}) THEN 1
        WHEN variant = 'control'
             AND RAND() < 0.10 THEN 1
        ELSE 0
      END AS converted,
      assigned_at
    FROM assigned
    """
    bq_client.query(sql).result()
    return f"{user_project}.visql_synth.{EXPERIMENT_TABLE_NAME}"

# ════════════════════════════════════════════════════════════════════
# PRE-EXECUTION TABLE-EXISTENCE CHECK (slide 4)
# ════════════════════════════════════════════════════════════════════
def check_tables_exist(sql: str, schema: DatabaseSchema) -> tuple[bool, list[str]]:
    """Catch hallucinated table names BEFORE the BigQuery round trip.

    Returns (all_ok, [missing_table_names]).
    """
    # Allow-list: known table names + their fq_names
    valid = set()
    for t in schema.tables:
        valid.add(t.name.lower())
        if t.fq_name:
            valid.add(t.fq_name.lower())
            # also allow the bare last segment
            valid.add(t.fq_name.split(".")[-1].lower())

    # Extract every FROM/JOIN target from the SQL
    pattern = re.compile(r"(?i)\b(?:FROM|JOIN)\s+`?([\w\-\.]+)`?", re.IGNORECASE)
    referenced = pattern.findall(sql)
    missing = []
    for r in referenced:
        rl = r.lower()
        bare = rl.split(".")[-1]
        # wildcard tables like ga_sessions_* are allowed
        if bare.endswith("_*"):
            continue
        if rl in valid or bare in valid:
            continue
        missing.append(r)
    return (len(missing) == 0), missing

# ════════════════════════════════════════════════════════════════════
# DATABASE MANAGER — wraps BQ client with the safety checks
# ════════════════════════════════════════════════════════════════════
class DatabaseManager:
    """Wraps a BigQuery client with table-existence + wildcard handling."""
    def __init__(self, schema: DatabaseSchema, bq_client=None):
        self.schema = schema
        self.bq = bq_client

    def execute(self, sql: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """Returns (dataframe, error_message). On success, error is None."""
        if self.bq is None:
            return None, "DatabaseManager has no bq_client attached."

        # Wildcard-table fixup for GA sharded tables
        sql = self._fix_ga_wildcards(sql)

        try:
            df = self.bq.query(sql).to_dataframe()
            return df, None
        except Exception as e:
            return None, str(e)

    def _fix_ga_wildcards(self, sql: str) -> str:
        r"""`FROM ga_sessions_*` -> `FROM \`bigquery-public-data.google_analytics_sample.ga_sessions_*\``"""
        if (self.schema.bq_project and self.schema.bq_dataset
                and self.schema.db_id.startswith("ga")):
            prefix = f"`{self.schema.bq_project}.{self.schema.bq_dataset}."

            def _rewrite(m: re.Match) -> str:
                kw, tbl = m.group(1), m.group(2)
                return f"{kw} {prefix}{tbl}`"
            sql = re.sub(
                r"(?i)\b(FROM|JOIN)\s+([a-zA-Z_][\w]*_\*)\b(?!\s*`)",
                _rewrite, sql,
            )
        return sql
