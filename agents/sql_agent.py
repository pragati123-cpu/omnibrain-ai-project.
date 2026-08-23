"""
sql_agent.py
-------------
Text-to-SQL Agent: handles queries about historical stock data.

Per the project spec, this is scoped to EXTERNAL historical stock
data -- a separate structured source from the PDF (see
ingestion/README.md's "Design decisions" for why PDF tables are
handled by the Vision Agent instead, not this one).

Since no real market-data feed is wired up yet, this module also
builds a small SAMPLE SQLite database of synthetic daily stock
prices, so the agent has something real to query during development.
Swap `build_sample_db()` for a real data loader (stock API/CSV) once
the team has a real source -- the rest of the agent doesn't change,
since it only depends on the schema, not where the data came from.

Flow:
  natural language question
    -> GPT-4o translates it into a single SQL SELECT statement
       against the documented schema
    -> the statement is validated as read-only (SELECT only)
    -> executed against SQLite
    -> results returned

Requires: OPENAI_API_KEY set in the environment.
"""

import random
import re
import sqlite3
from datetime import date, timedelta

from openai import OpenAI

from config import STOCK_DB_PATH, OPENAI_MODEL_SQL

_openai_client = None

SCHEMA_DESCRIPTION = """
Table: stock_prices
Columns:
  date    TEXT     (format YYYY-MM-DD)
  ticker  TEXT     (stock ticker symbol, e.g. 'ACME')
  open    REAL
  high    REAL
  low     REAL
  close   REAL
  volume  INTEGER
"""


def _get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def build_sample_db(force: bool = False):
    """
    Creates a small synthetic daily stock price dataset for ticker
    'ACME' across fiscal year 2025 (~1 trading year, weekends
    skipped), so the SQL Agent has real data to query during
    development. Idempotent unless force=True.
    """
    STOCK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if STOCK_DB_PATH.exists() and not force:
        return

    conn = sqlite3.connect(STOCK_DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS stock_prices")
    cur.execute(
        """
        CREATE TABLE stock_prices (
            date TEXT,
            ticker TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
        """
    )

    random.seed(42)  # reproducible sample data across runs
    start = date(2025, 1, 1)
    price = 120.0

    rows = []
    for i in range(365):
        day = start + timedelta(days=i)
        if day.weekday() >= 5:  # skip weekends
            continue
        drift_pct = random.gauss(0.05, 1.2) / 100  # slight upward drift
        open_p = price
        close_p = max(1.0, open_p * (1 + drift_pct))
        high_p = max(open_p, close_p) * (1 + random.uniform(0, 0.01))
        low_p = min(open_p, close_p) * (1 - random.uniform(0, 0.01))
        volume = random.randint(800_000, 2_500_000)

        rows.append(
            (
                day.isoformat(),
                "ACME",
                round(open_p, 2),
                round(high_p, 2),
                round(low_p, 2),
                round(close_p, 2),
                volume,
            )
        )
        price = close_p

    cur.executemany(
        "INSERT INTO stock_prices VALUES (?, ?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    conn.close()
    print(f"Sample stock database created at {STOCK_DB_PATH} ({len(rows)} rows)")


def _generate_sql(query: str) -> str:
    openai_client = _get_openai()
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL_SQL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You translate natural language questions into a single SQLite "
                    "SELECT statement. Output ONLY the raw SQL -- no explanation, no "
                    "markdown code fences. Only SELECT statements are permitted.\n\n"
                    f"Schema:\n{SCHEMA_DESCRIPTION}"
                ),
            },
            {"role": "user", "content": query},
        ],
        max_tokens=200,
        temperature=0,
    )
    sql = response.choices[0].message.content.strip()
    sql = re.sub(r"^```sql|```$", "", sql, flags=re.IGNORECASE).strip()
    return sql


def _is_safe_select(sql: str) -> bool:
    """
    Guardrail: only allow a single, read-only SELECT statement.
    Rejects anything containing write/DDL keywords or a second
    statement (via a semicolon), even if it starts with SELECT.
    """
    normalized = sql.strip().rstrip(";").upper()
    if not normalized.startswith("SELECT"):
        return False
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "ATTACH", "PRAGMA", ";"]
    return not any(word in normalized for word in forbidden)


def sql_agent(query: str) -> dict:
    build_sample_db()  # no-op if it already exists

    generated_sql = _generate_sql(query)

    if not _is_safe_select(generated_sql):
        return {
            "agent": "sql",
            "query": query,
            "generated_sql": generated_sql,
            "error": "Generated SQL failed the read-only SELECT safety check; refused to execute.",
            "results": [],
            "citations": [],
        }

    conn = sqlite3.connect(STOCK_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute(generated_sql)
        rows = [dict(r) for r in cur.fetchall()]
    except sqlite3.Error as e:
        conn.close()
        return {
            "agent": "sql",
            "query": query,
            "generated_sql": generated_sql,
            "error": f"SQL execution failed: {e}",
            "results": [],
            "citations": [],
        }
    conn.close()

    return {
        "agent": "sql",
        "query": query,
        "generated_sql": generated_sql,
        "results": rows,
        "citations": [{"source": "historical_stock_prices.db (sample/synthetic data)"}],
    }


if __name__ == "__main__":
    import json
    import sys

    build_sample_db()
    query = sys.argv[1] if len(sys.argv) > 1 else "What was the closing price on 2025-03-14?"
    print(json.dumps(sql_agent(query), indent=2))
