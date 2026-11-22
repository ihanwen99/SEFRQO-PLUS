import re

import psycopg2
from psycopg2.extras import RealDictCursor

from sefrqo_plus import config

STATEMENT_TIMEOUT_MS = 60000
MIN_REWARD = -3


def extract_execution_time(explain_analyze_lines):
    for line in reversed(explain_analyze_lines):
        if "Execution Time" in line:
            try:
                return float(
                    line.strip().split("Execution Time:")[1].strip().split(" ")[0]
                )
            except (IndexError, ValueError) as e:
                print(f"Error extracting execution time: {e}")
                return None
    return None


def measure_query_latency(sql_query):
    try:
        conn = psycopg2.connect(
            host=config.PG_HOST,
            port=config.PG_PORT,
            user=config.PG_USER,
            dbname="imdbload",
        )
    except psycopg2.Error as e:
        print(f"Database Connection Error: {e}")
        return None
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS};")
            cursor.execute(sql_query)
            cursor.fetchall()
            cursor.execute(f"EXPLAIN ANALYZE {sql_query}")
            lines = [row["QUERY PLAN"] for row in cursor.fetchall()]
        conn.commit()
        return extract_execution_time(lines)
    except psycopg2.Error as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        conn.close()


def compute_latency_reward(llm_query_latency, baseline_latency):
    return (baseline_latency - llm_query_latency) / baseline_latency


def latency_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    for i in range(len(completions)):
        match = re.search(
            r"\*\*\*\s*sql\s*\*\*\*\s*([\s\S]*?);", prompts[i][0]["content"]
        )
        sql = match.group(1).strip()
        hint = "/*+ {} */".format(completions[i][0]["content"])
        current_execution_time = measure_query_latency(hint + sql)
        if current_execution_time is None:
            rewards.append(MIN_REWARD)
            continue
        reward = compute_latency_reward(current_execution_time, int(answer[i]))
        if reward < MIN_REWARD:
            reward = MIN_REWARD
        rewards.append(reward)
    return rewards
