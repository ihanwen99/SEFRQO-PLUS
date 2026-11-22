import re
import time

import pandas as pd
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from .encoders import create_query_encoder
from .postgres import execute_sql

EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"
HISTORY_LIMIT = 1000
INIT = "sql_init_collection"
ONLINE = "sql_online_collection"
POSTGRES = "sql_postgres_collection"
QUERY_LIMIT = 10000
KEEP_ALL_BELOW = 1000

HISTORY_ENTRY = re.compile(
    r"^\s*--\s*(\S+)\s*\n\s*--\s*([\d.]+)\s*\n\s*(/\*[\s\S]*?\*/)\s*(.+)$", re.DOTALL
)


def add_explain_analyze(sql):
    return sql if "EXPLAIN" in sql.upper() else "EXPLAIN (ANALYZE, FORMAT JSON) " + sql


class RecordStore:
    def __init__(self, path, encoder="text"):
        self.text_model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = MilvusClient(path)
        self.encode, self.dim = create_query_encoder(encoder, self.text_model)

    def create_collections(self, history_files):
        if INIT not in self.client.list_collections():
            print(f"Collection '{INIT}' does not exist, initializing...")
            self.client.create_collection(collection_name=INIT, dimension=self.dim)
            data = []
            for path in history_files:
                with open(path, "r", encoding="utf-8") as file:
                    data.extend(file.read().split("\n\n"))
            sql_plan_pairs = []
            for entry in data:
                entry = entry.strip()
                if not entry:
                    continue
                match = HISTORY_ENTRY.match(entry)
                if match:
                    sql_plan_pairs.append(
                        (
                            match.group(1).strip(),
                            match.group(4).strip(),
                            match.group(3).strip(),
                            float(match.group(2).strip()),
                        )
                    )
                else:
                    print("Warning: entry does not match expected format, skipping:")
                    print(entry)
            for counter, (file_id, sql, plan, exec_time) in enumerate(sql_plan_pairs):
                self.client.insert(
                    collection_name=INIT,
                    data=[
                        {
                            "id": counter,
                            "iteration": 0,
                            "sqlid": file_id,
                            "vector": self.encode(sql),
                            "sql": sql,
                            "plan": plan,
                            "execution_time": exec_time,
                        }
                    ],
                )
            print(f"Milvus collection '{INIT}' initialized.")
        else:
            print(f"Collection '{INIT}' already exists.")

        for name in (ONLINE, POSTGRES):
            if name not in self.client.list_collections():
                print(f"Collection '{name}' does not exist, initializing...")
                self.client.create_collection(collection_name=name, dimension=self.dim)
                print(f"Milvus collection '{name}' initialized.")
            else:
                print(f"Collection '{name}' already exists.")

    def search_references(self, vector, sqlid, limit):
        start = time.perf_counter()
        res = self.client.search(
            collection_name=INIT,
            data=[vector],
            limit=limit,
            output_fields=["sql", "plan", "sqlid"],
            filter=f"sqlid != '{sqlid}'",
        )
        self.last_search_ms = round(float((time.perf_counter() - start) * 1000.0), 3)
        return [r["entity"] for r in res[0]]

    def insert_record(self, collection, record):
        self.client.insert(collection_name=collection, data=[record])

    def fill_postgres_collection(self, keys, values, database, timeout_ms):
        print("Starting first round of SQL execution (cache warm-up)...")
        for sql in values:
            execute_sql(add_explain_analyze(sql), database, timeout_ms)
        print("First round of SQL execution completed.")
        time.sleep(1)

        print("Starting second round of SQL execution and collecting results...")
        counter = 0
        for file_name, sql in zip(keys, values):
            vector = self.encode(sql)
            exec_time, plan = execute_sql(
                add_explain_analyze(sql), database, timeout_ms
            )
            if exec_time is None or plan is None:
                print(
                    f"Warning: SQL {file_name} did not return a valid execution time or plan, skipping."
                )
                continue
            self.insert_record(
                POSTGRES,
                {
                    "id": counter,
                    "iteration": 1,
                    "sqlid": file_name,
                    "vector": vector,
                    "sql": sql,
                    "plan": plan,
                    "execution_time": exec_time,
                },
            )
            print(
                f"SQL {file_name} (execution time {exec_time} ms) inserted into collection '{POSTGRES}'."
            )
            counter += 1
        print("Postgres collection filled successfully.")

    def fill_postgres_collection_one_record(self, key, value, database, timeout_ms):
        print("Starting first round of SQL execution (cache warm-up)...")
        execute_sql(add_explain_analyze(value), database, timeout_ms)
        print("First round of SQL execution completed.")
        time.sleep(1)

        print("Starting second round of SQL execution and collecting results...")
        vector = self.encode(value)
        exec_time, plan = execute_sql(add_explain_analyze(value), database, timeout_ms)
        if exec_time is None or plan is None:
            print(
                f"Warning: SQL {key} did not return a valid execution time or plan, skipping."
            )
            return
        new_id = self.get_next_id(POSTGRES)
        self.insert_record(
            POSTGRES,
            {
                "id": new_id,
                "iteration": 1,
                "sqlid": key,
                "vector": vector,
                "sql": value,
                "plan": plan,
                "execution_time": exec_time,
            },
        )
        print(
            f"A new SQL {key} (execution time {exec_time} ms) inserted into collection '{POSTGRES}' with ID {new_id}."
        )
        print("Postgres collection filled a new query successfully.")

    def fetch_all_records(self, collection):
        return self.client.query(
            collection_name=collection, filter="", limit=QUERY_LIMIT
        )

    def find_fastest_plan(self, collection, sqlid):
        records = self.client.query(
            collection_name=collection, filter=f"sqlid == '{sqlid}'", limit=QUERY_LIMIT
        )
        if not records:
            return "NOT FOUND", 0
        fastest = min(records, key=lambda r: r["execution_time"])
        return fastest["plan"], fastest["execution_time"]

    def keep_fastest_records(self, collection, keep_all=False):
        records = self.fetch_all_records(collection)
        if not records:
            return False
        if len(records) <= KEEP_ALL_BELOW:
            return True
        if keep_all:
            return True
        records = [{k: v for k, v in r.items() if k != "vector"} for r in records]
        df = pd.DataFrame(records)
        if "sqlid" not in df.columns or "execution_time" not in df.columns:
            return False
        fastest = df.loc[df.groupby("sqlid")["execution_time"].idxmin()]
        ids_to_keep = set(fastest["id"].tolist())
        ids_to_delete = [r["id"] for r in records if r["id"] not in ids_to_keep]
        if ids_to_delete:
            self.client.delete(collection_name=collection, ids=ids_to_delete)
        return True

    def update_references(self):
        postgres_records = [
            {k: v for k, v in r.items() if k != "vector"}
            for r in self.fetch_all_records(POSTGRES)
        ]
        online_records = [r.copy() for r in self.fetch_all_records(ONLINE)]
        if not postgres_records or not online_records:
            print("One of the collections has no records.")
            return False
        df_postgres = pd.DataFrame(postgres_records)
        df_online = pd.DataFrame(online_records)
        for df, name in [(df_postgres, "Postgres"), (df_online, "Online")]:
            if "sqlid" not in df.columns or "execution_time" not in df.columns:
                print(
                    f"{name} records are missing required fields 'sqlid' or 'execution_time'."
                )
                return False

        postgres_fastest = {
            row["sqlid"]: row.to_dict()
            for _, row in df_postgres.loc[
                df_postgres.groupby("sqlid")["execution_time"].idxmin()
            ].iterrows()
        }
        online_fastest = {
            row["sqlid"]: row.to_dict()
            for _, row in df_online.loc[
                df_online.groupby("sqlid")["execution_time"].idxmin()
            ].iterrows()
        }

        records_to_add = []
        for sqlid, online_rec in online_fastest.items():
            pg = postgres_fastest.get(sqlid)
            if (
                pg is not None
                and online_rec["execution_time"] < pg["execution_time"]
                and online_rec["plan"].strip() != pg["plan"].strip()
            ):
                records_to_add.append(online_rec)

        if records_to_add:
            first_id = self.get_next_id(INIT)
            for i, record in enumerate(records_to_add):
                self.insert_record(INIT, {**record, "id": first_id + i})
            print(f"Inserted {len(records_to_add)} new online records into {INIT}.")
        else:
            print("No new records qualified for insertion.")

        if not self.keep_fastest_records(INIT):
            print("keep_fastest_records failed.")
            return False
        return True

    def find_online_records(self, sqlid, iteration=None):
        flt = f"sqlid == '{sqlid}'" + (
            "" if iteration is None else f" and iteration == {iteration}"
        )
        try:
            return self.client.query(
                collection_name=ONLINE, filter=flt, limit=HISTORY_LIMIT
            )
        except Exception:
            return []

    def get_next_id(self, collection):
        records = self.fetch_all_records(collection)
        ids = [r["id"] for r in records if "id" in r]
        return max(ids) + 1 if ids else 0
