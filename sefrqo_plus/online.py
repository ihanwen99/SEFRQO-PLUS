import json
import math
import os
import random
import time

from . import prompts
from .postgres import (
    execute_sql,
    extract_hint_block,
    get_postgres_hint,
    format_query_statistics,
)
from .records import ONLINE, POSTGRES

TIMING_KEYS = [
    "rag_query_encode_ms",
    "rag_search_ms",
    "rag_reference_context_ms",
    "rag_latency_ms",
    "query_statistics_ms",
    "history_lookup_ms",
    "prompt_build_ms",
    "llm_inference_ms",
    "online_overhead_ms",
]


def read_sql_files(directory, sort=False):
    sql_files = {}
    names = os.listdir(directory)
    for filename in sorted(names) if sort else names:
        if filename.endswith(".sql"):
            with open(os.path.join(directory, filename), "r") as file:
                sql_files[filename] = file.read()
    return sql_files


def elapsed_ms(start):
    return round(float((time.perf_counter() - start) * 1000.0), 3)


class OnlineOptimizer:
    def __init__(self, cfg, store, generate):
        self.cfg = cfg
        self.store = store
        self.generate = generate
        self.online_counter = 0
        self.plus = cfg.get("prompt_style", "sefrqo") == "plus"
        self.reseed = cfg.get("reseed_statistics_rng", False)
        self.run_started_at = time.strftime("%Y%m%d_%H%M%S")
        self.overhead_log = None
        if cfg.get("overhead_log"):
            self.overhead_log = os.path.join(
                cfg["overhead_log"],
                f"{self.run_started_at}_{cfg['name']}_overhead.jsonl",
            )

    def compute_statistics(self, sql, database):
        return format_query_statistics(sql, database, reseed_rng=self.reseed)

    def find_reference_database(self, sqlid):
        if len(self.cfg["databases"]) == 1:
            return self.cfg["databases"][0]["name"]
        for db in self.cfg["databases"]:
            if db["name"] in sqlid:
                return db["name"]
        raise ValueError(f"Invalid database name in result['entity']['sqlid']: {sqlid}")

    def build_sefrqo_prompt(
        self, key, sql, database, timeout_ms, question_stats, first_iteration, refs
    ):
        cfg = self.cfg
        if first_iteration:
            pg_hint = get_postgres_hint(sql, database)
            compact = (
                cfg.get("compact_first_instruction", False) and cfg["use_references"]
            )
            user = prompts.build_first_prompt(question_stats, pg_hint, compact)
        else:
            best_hint, best_time = self.store.find_fastest_plan(ONLINE, key)
            if best_hint is None:
                best_hint = "NOT FOUND"
            _, postgres_time = self.store.find_fastest_plan(POSTGRES, key)
            if postgres_time == 0:
                self.store.fill_postgres_collection_one_record(
                    key, sql, database, timeout_ms
                )
                _, postgres_time = self.store.find_fastest_plan(POSTGRES, key)
            pg_hint = get_postgres_hint(sql, database)
            user = prompts.build_feedback_prompt(
                question_stats,
                pg_hint,
                best_hint,
                prompts.format_gain(postgres_time, best_time),
                cfg["use_references"],
            )
        if cfg["use_references"]:
            user = prompts.build_references_section(cfg["num_references"], *refs) + user
        return user

    def build_plus_prompt(
        self, key, question_stats, iteration, first_iteration, refs, timing
    ):
        cfg = self.cfg
        clamp = cfg.get("clamp_references", False)
        if first_iteration:
            user = prompts.build_plus_first_prompt(question_stats)
            verb = "is"
        else:
            start = time.perf_counter()
            best_hint, best_time = self.store.find_fastest_plan(ONLINE, key)
            if best_hint is None:
                best_hint = "NOT FOUND"
            first_times = [
                r.get("execution_time") for r in self.store.find_online_records(key, 2)
            ]
            first_times = [t for t in first_times if prompts.is_valid_time(t)]
            first_time = min(first_times) if first_times else None
            recent = self.find_recent_record(key, iteration)
            recent_hint = recent.get("plan", "NOT FOUND") if recent else "NOT FOUND"
            recent_time = recent.get("execution_time", 0) if recent else 0
            timing["history_lookup_ms"] = elapsed_ms(start)
            user = prompts.build_plus_feedback_prompt(
                question_stats,
                best_hint,
                prompts.format_best_gain_line(first_time, best_time),
                recent_hint,
                prompts.format_recent_gain_line(first_time, recent_time),
            )
            verb = cfg.get("feedback_reference_verb", "is")
        if cfg["use_references"]:
            user = (
                prompts.build_plus_references_section(
                    cfg["num_references"], *refs, verb=verb, clamp=clamp
                )
                + user
            )
        return user

    def find_recent_record(self, key, iteration):
        if self.cfg.get("recent_record", "previous_iteration") == "latest":
            rows = self.store.find_online_records(key)
            if not rows:
                return None
            last = max(r.get("iteration", -1) for r in rows)
            rows = [r for r in rows if r.get("iteration", -1) == last]
        else:
            rows = self.store.find_online_records(key, iteration - 1)
            if not rows:
                return None
        return max(rows, key=lambda r: r.get("id", -1))

    def build_messages(
        self, key, sql, vector, database, timeout_ms, iteration, first_iteration, timing
    ):
        cfg = self.cfg
        start = time.perf_counter()
        reference_plans, reference_stats = [], []
        for ref in self.store.search_references(vector, key, cfg["num_references"]):
            reference_plans.append(ref["plan"])
            reference_stats.append(
                self.compute_statistics(
                    ref["sql"], self.find_reference_database(ref["sqlid"])
                )
            )
        timing["rag_reference_context_ms"] = elapsed_ms(start)

        start = time.perf_counter()
        with open(cfg["system_prompt"], "r") as file:
            system_prompt = file.read()
        messages = (
            [{"role": "system", "content": system_prompt}]
            if cfg["use_system_prompt"]
            else []
        )
        stats_start = time.perf_counter()
        question_stats = self.compute_statistics(sql, database)
        timing["query_statistics_ms"] = elapsed_ms(stats_start)
        refs = (reference_stats, reference_plans)
        if self.plus:
            user = self.build_plus_prompt(
                key, question_stats, iteration, first_iteration, refs, timing
            )
        else:
            user = self.build_sefrqo_prompt(
                key, sql, database, timeout_ms, question_stats, first_iteration, refs
            )
        messages.append({"role": "user", "content": user})
        timing["prompt_build_ms"] = elapsed_ms(start)
        return messages

    def output_to_hint(self, output):
        if self.cfg["hint_output"] == "body":
            return "/*+ " + output + " */"
        return extract_hint_block(output)

    def optimize_query(
        self, key, sql, database, timeout_ms, iteration, first_iteration, last_iteration
    ):
        print(f"=========Processing question {key}=========\n")
        timing = dict.fromkeys(TIMING_KEYS, 0.0)
        overhead_start = time.perf_counter()
        start = time.perf_counter()
        vector = self.store.encode(sql)
        timing["rag_query_encode_ms"] = elapsed_ms(start)
        messages = self.build_messages(
            key, sql, vector, database, timeout_ms, iteration, first_iteration, timing
        )
        timing["rag_search_ms"] = self.store.last_search_ms
        timing["rag_latency_ms"] = round(
            timing["rag_query_encode_ms"]
            + timing["rag_search_ms"]
            + timing["rag_reference_context_ms"],
            3,
        )
        start = time.perf_counter()
        output = self.generate(messages)
        timing["llm_inference_ms"] = elapsed_ms(start)
        timing["online_overhead_ms"] = elapsed_ms(overhead_start)
        hint = self.output_to_hint(output)

        print(f"hint for question {key} given by LLM (may not be valid): ", hint)
        print("\n")
        print(f"Executing sql query {key} with hints")
        exec_time, real_hint = execute_sql(
            hint + "\n" + "EXPLAIN (ANALYZE, FORMAT JSON) " + sql, database, timeout_ms
        )
        print("\n")

        record = {
            "id": self.online_counter,
            "iteration": iteration,
            "sqlid": key,
            "vector": vector,
            "sql": sql,
            "plan": real_hint,
            "execution_time": exec_time,
        }
        if self.overhead_log:
            record.update(
                {
                    k: timing[k]
                    for k in (
                        "llm_inference_ms",
                        "rag_latency_ms",
                        "online_overhead_ms",
                    )
                }
            )
        self.store.insert_record(ONLINE, record)
        self.online_counter += 1
        print(
            f"Inserted online execution result for question {key} into Milvus collection '{ONLINE}'.\n"
        )
        self.print_report(
            key, sql, database, timeout_ms, iteration, last_iteration, exec_time, timing
        )

    def log_overhead(
        self,
        key,
        database,
        timeout_ms,
        iteration,
        postgres_time,
        exec_time,
        best_time,
        timing,
    ):
        net_time_ms = round(float(exec_time + timing["online_overhead_ms"]), 3)
        gross = net = None
        if isinstance(postgres_time, (int, float)) and postgres_time > 0:
            gross = round((postgres_time - exec_time) / postgres_time * 100.0, 3)
            net = round((postgres_time - net_time_ms) / postgres_time * 100.0, 3)
        record = {
            "run_started_at": self.run_started_at,
            "workload": self.cfg["workload"],
            "workload_label": self.cfg["workload"].upper(),
            "database": database,
            "encoder": self.cfg.get("encoder", "text"),
            "k": self.cfg["num_references"],
            "timeout_ms": timeout_ms,
            "iteration": iteration,
            "sqlid": key,
            "postgres_time_ms": postgres_time,
            "execution_time_ms": exec_time,
            "net_time_ms": net_time_ms,
            "gross_gain_pct": gross,
            "net_gain_pct": net,
            "best_online_time_ms": best_time,
            "timing": timing,
        }
        with open(self.overhead_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"Online overhead for question {key}: RAG={timing['rag_latency_ms']} ms, "
            f"LLM={timing['llm_inference_ms']} ms, total={timing['online_overhead_ms']} ms"
        )

    def print_report(
        self,
        key,
        sql,
        database,
        timeout_ms,
        iteration,
        last_iteration,
        exec_time,
        timing,
    ):
        _, postgres_time = self.store.find_fastest_plan(POSTGRES, key)
        if postgres_time == 0:
            self.store.fill_postgres_collection_one_record(
                key, sql, database, timeout_ms
            )
            _, postgres_time = self.store.find_fastest_plan(POSTGRES, key)
        pg_hint = get_postgres_hint(sql, database)
        best_hint, best_time = self.store.find_fastest_plan(ONLINE, key)
        if self.overhead_log:
            self.log_overhead(
                key,
                database,
                timeout_ms,
                iteration,
                postgres_time,
                exec_time,
                best_time,
                timing,
            )
        print(
            f"Postgres execution plan ({postgres_time} ms) for question {key} is \n{pg_hint}\n"
        )
        gain = (
            "UNKNOWN"
            if postgres_time == 0
            else f"{postgres_time - best_time} / {postgres_time} = {((postgres_time - best_time) / postgres_time * 100):.3f}%"
        )
        if last_iteration:
            print(
                f"Final LLM generated optimal execution plan ({best_time} ms) for question {key} is \n{best_hint}\n"
            )
            print(f"The final performance gain is {gain}\n")
        else:
            if best_time == 0:
                print(f"The current LLM generated optimal execution plan is UNKNOWN\n")
            else:
                print(
                    f"Current LLM generated optimal execution plan ({best_time} ms) for question {key} is \n{best_hint}\n"
                )
            print(f"The current performance gain is {gain}\n")
        if (
            best_hint is None
            or best_hint == "NOT FOUND"
            or pg_hint.strip() == best_hint.strip()
        ):
            print(
                f"The final/current LLM generated optimal execution plan for question {key} is the same as the Postgres execution plan\n"
            )
        else:
            print(
                f"The final/current LLM generated optimal execution plan for question {key} is different from the Postgres execution plan\n"
            )

    def end_iteration(self):
        self.store.keep_fastest_records(
            ONLINE, keep_all=self.cfg.get("keep_all_online", False)
        )
        self.store.update_references()

    def run_static(self):
        db = self.cfg["databases"][0]
        workload = read_sql_files(
            db["workloads"][0], self.cfg.get("sort_workload", False)
        )
        keys, values = list(workload.keys()), list(workload.values())
        assert len(keys) > 0, "The number of questions should be greater than 0"
        if self.cfg.get("continue_online_ids"):
            self.online_counter = self.store.get_next_id(ONLINE)
        n = self.cfg["num_iterations"]
        for it in range(n):
            if it == 0:
                self.store.fill_postgres_collection(
                    keys, values, db["name"], db["timeout_ms"]
                )
                continue
            print(f"=========Processing iteration {it + 1}=========\n")
            for key, sql in zip(keys, values):
                self.optimize_query(
                    key, sql, db["name"], db["timeout_ms"], it + 1, it == 1, it == n - 1
                )
            self.end_iteration()

    def run_dynamic(self):
        n = self.cfg["num_iterations"]
        episodes = partition_randomly(n, self.cfg["num_episodes"])
        global_iter = 0
        for length in episodes:
            db = random.choice(self.cfg["databases"])
            directory = random.choice(db["workloads"])
            workload = read_sql_files(directory, self.cfg.get("sort_workload", False))
            keys, values = list(workload.keys()), list(workload.values())
            assert len(keys) > 0, "The number of questions should be greater than 0"
            for local_iter in range(length):
                if global_iter == 0:
                    self.store.fill_postgres_collection(
                        keys, values, db["name"], db["timeout_ms"]
                    )
                    global_iter += 1
                    continue
                print(
                    f"=========Processing global iteration {global_iter + 1}/{n}=========\n"
                )
                print(
                    f"=========Processing local iteration {local_iter + 1}/{length}=========\n"
                )
                for key, sql in zip(keys, values):
                    self.optimize_query(
                        key,
                        sql,
                        db["name"],
                        db["timeout_ms"],
                        global_iter + 1,
                        global_iter == 1,
                        global_iter == n - 1,
                    )
                self.end_iteration()
                global_iter += 1


def partition_randomly(total, parts):
    lower_bound = math.ceil(total / parts / 2)
    upper_bound = math.floor(2 * total / parts)
    result = []
    remaining_total, remaining_parts = total, parts
    for _ in range(parts):
        current_lower = max(
            lower_bound, remaining_total - (remaining_parts - 1) * upper_bound
        )
        current_upper = min(
            upper_bound, remaining_total - (remaining_parts - 1) * lower_bound
        )
        value = random.randint(current_lower, current_upper)
        result.append(value)
        remaining_total -= value
        remaining_parts -= 1
    random.shuffle(result)
    return result
