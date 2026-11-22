import json
import random
import re

import psycopg2

from . import config


def connect(database):
    return psycopg2.connect(
        host=config.PG_HOST, port=config.PG_PORT, user=config.PG_USER, dbname=database
    )


def execute_sql(sql_query, database, timeout_ms):
    exec_time = None
    bracket_plan = None
    connection = None
    cursor = None
    try:
        connection = connect(database)
        connection.autocommit = True
        cursor = connection.cursor()
        cursor.execute(f"SET statement_timeout = {timeout_ms};")
        cursor.execute("load 'pg_hint_plan';")
        cursor.execute(sql_query)
        result = cursor.fetchone()
        cursor.execute("DISCARD ALL;")

        if result is not None:
            json_data = result[0]
            if isinstance(json_data, str):
                json_data = json.loads(json_data)
            if isinstance(json_data, list) and len(json_data) > 0:
                exec_time = json_data[0].get("Execution Time")
                bracket_plan = plan_to_bracket_hint(json.dumps(json_data[0]))
            else:
                print("Unexpected JSON structure:", json_data)
        else:
            print("No result returned from query execution")
    except psycopg2.extensions.QueryCanceledError as e:
        print(f"Query timed out: {e}")
        exec_time = timeout_ms
    except Exception as error:
        print(f"Error: {error}")
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

    if exec_time is None:
        exec_time = timeout_ms
    return exec_time, bracket_plan


def explain_plan_json(sql, database):
    with connect(database) as conn:
        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN (FORMAT JSON) {sql};")
            result = cur.fetchall()
            plan = result[0][0]
    return json.dumps(plan[0])


def get_postgres_hint(sql, database):
    return plan_to_bracket_hint(explain_plan_json(sql, database))


def extract_join_tree(plan):
    node_type = plan.get("Node Type", "")

    if node_type in ("Nested Loop", "Hash Join", "Merge Join"):
        plans = plan.get("Plans", [])
        if len(plans) >= 2:
            left_tree = extract_join_tree(plans[0])
            right_tree = extract_join_tree(plans[1])
            return {
                "type": "join",
                "op": node_type.replace(" ", ""),
                "left": left_tree,
                "right": right_tree,
            }
        elif "Plans" in plan and plan["Plans"]:
            return extract_join_tree(plan["Plans"][0])
        else:
            return None
    elif "Scan" in node_type:
        alias = plan.get("Alias") or plan.get("Relation Name") or ""
        return {"type": "scan", "alias": alias, "scan_op": node_type.replace(" ", "")}
    elif "Plans" in plan:
        child_trees = [extract_join_tree(child) for child in plan["Plans"]]
        child_trees = [t for t in child_trees if t is not None]
        if not child_trees:
            return None
        if len(child_trees) == 1:
            return child_trees[0]
        tree = child_trees[0]
        for child in child_trees[1:]:
            tree = {"type": "join", "op": "DummyJoin", "left": tree, "right": child}
        return tree
    return None


def flatten_join_tree(tree):
    if tree is None:
        return []
    if tree["type"] == "scan":
        return [tree["alias"]]
    elif tree["type"] == "join":
        return flatten_join_tree(tree["left"]) + flatten_join_tree(tree["right"])
    return []


def collect_join_nodes(tree):
    if tree is None or tree["type"] == "scan":
        return []
    nodes = []
    nodes.extend(collect_join_nodes(tree["left"]))
    nodes.extend(collect_join_nodes(tree["right"]))
    nodes.append((tree["op"], flatten_join_tree(tree)))
    return nodes


def collect_scans(tree):
    if tree is None:
        return []
    if tree["type"] == "scan":
        return [(tree["scan_op"], tree["alias"])]
    elif tree["type"] == "join":
        return collect_scans(tree["left"]) + collect_scans(tree["right"])
    return []


def build_leading_string(tree):
    if tree is None:
        return ""
    if tree["type"] == "scan":
        return tree["alias"]
    elif tree["type"] == "join":
        left_str = build_leading_string(tree["left"])
        right_str = build_leading_string(tree["right"])
        if tree["op"] == "DummyJoin":
            return left_str + " " + right_str
        return f"({left_str} {right_str})"
    return ""


def plan_to_bracket_hint(plan_json_str):
    plan_data = json.loads(plan_json_str)
    root_plan = plan_data["Plan"] if "Plan" in plan_data else plan_data
    join_tree = extract_join_tree(root_plan)

    lines = []
    if join_tree is not None:
        for op, aliases in reversed(collect_join_nodes(join_tree)):
            if op == "DummyJoin":
                continue
            lines.append(f"{op}(" + " ".join(aliases) + ")")
        for scan_op, alias in collect_scans(join_tree):
            lines.append(f"{scan_op}({alias})")
        lines.append(f"Leading({build_leading_string(join_tree)})")
    elif "Scan" in root_plan.get("Node Type", ""):
        alias = root_plan.get("Alias") or root_plan.get("Relation Name") or ""
        op = root_plan.get("Node Type").replace(" ", "")
        lines.append(f"{op}({alias})")
        lines.append(f"Leading({alias})")
    return "/*+ " + "\n ".join(lines) + " */"


def extract_plan_cardinalities(plan_json):
    table_cardinality = {}
    filter_cardinality = {}

    def traverse(node):
        if not isinstance(node, dict):
            return
        if "Relation Name" in node:
            alias = node.get("Alias", node["Relation Name"])
        elif "CTE Name" in node:
            alias = node.get("Alias", node["CTE Name"])
        else:
            alias = node.get("Alias", "unknown")
        plan_rows = node.get("Plan Rows")

        if "Filter" in node and plan_rows is not None:
            filter_cardinality[(alias, node["Filter"])] = plan_rows
        if "Join Filter" in node:
            join_filter_str = node["Join Filter"]
            aliases = re.findall(r"([a-zA-Z_]\w*)\.", join_filter_str) or [alias]
            for a in aliases:
                if plan_rows is not None:
                    filter_cardinality[(a, join_filter_str)] = plan_rows
        if "Relation Name" in node and plan_rows is not None:
            table_cardinality[alias] = max(plan_rows, table_cardinality.get(alias, 0))

        for value in node.values():
            if isinstance(value, dict):
                traverse(value)
            elif isinstance(value, list):
                for item in value:
                    traverse(item)

    traverse(plan_json["Plan"] if "Plan" in plan_json else plan_json)
    return table_cardinality, filter_cardinality


def format_query_statistics(sql, database, reseed_rng=False):
    if reseed_rng:
        random.seed(42)
    table_card, filter_card = extract_plan_cardinalities(
        json.loads(explain_plan_json(sql, database))
    )
    result_lines = ["Table Cardinality:"]
    for table, rows in table_card.items():
        result_lines.append(f"  {table}: {rows}")
    result_lines.append("Filter Cardinality:")
    for (alias, filt), rows in filter_card.items():
        result_lines.append(f"  ({alias}, {filt}): {rows}")
    return "\n".join(result_lines)


def extract_hint_block(text):
    match = re.search(r"(\/\*\+.*?\*\/)", text, re.DOTALL)
    if match:
        return match.group(1)
    return " "
