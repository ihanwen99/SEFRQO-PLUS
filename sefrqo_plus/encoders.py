import re
from typing import Dict, List, Tuple

import numpy as np
import sqlglot

SQL_KEYWORDS = set(
    """select from where join left right inner outer on group by order having limit union distinct exists in and or not like between case when then else end as is null true false
count sum avg min max create table index primary key foreign references insert update delete values into set alter add drop view materialized with""".split()
)

_ID = r"[A-Za-z_][\w$]*"


def _is_identifier(tok: str) -> bool:
    return re.fullmatch(_ID, tok) is not None


def _tokenize_sql(sql: str) -> List[str]:
    toks = re.findall(rf"{_ID}|\d+|<=|>=|<>|!=|==|[-+*/%(),.=<>]", sql)
    return [t for t in toks if t.strip() != ""]


def _read_dotted_identifier(tokens: List[str], i: int) -> Tuple[List[str], int]:
    parts = []
    j = i
    if j < len(tokens) and _is_identifier(tokens[j]):
        parts.append(tokens[j].lower())
        j += 1
        while (
            j + 1 < len(tokens) and tokens[j] == "." and _is_identifier(tokens[j + 1])
        ):
            j += 1
            parts.append(tokens[j].lower())
            j += 1
    return parts, j


def anonymize_sql(sql: str) -> str:
    tokens = _tokenize_sql(sql)
    table_map: Dict[str, str] = {}
    alias_to_base: Dict[str, str] = {}
    column_map: Dict[str, str] = {}

    def table_id(name: str) -> str:
        if name not in table_map:
            table_map[name] = f"T{len(table_map)}"
        return table_map[name]

    def column_id(name: str) -> str:
        if name not in column_map:
            column_map[name] = f"C{len(column_map)}"
        return column_map[name]

    out: List[str] = []
    last_kw = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = tok.lower() if _is_identifier(tok) else tok

        if _is_identifier(tok) and low in SQL_KEYWORDS:
            out.append(low)
            last_kw = low
            i += 1
            continue

        if last_kw in {"from", "join", "update", "into"} or (
            len(out) > 0 and out[-1] == "," and last_kw in {"from", "join"}
        ):
            parts, jnext = _read_dotted_identifier(tokens, i)
            if parts:
                base = parts[-1]
                out.append(table_id(base))
                k = jnext
                if k < len(tokens):
                    if _is_identifier(tokens[k]) and tokens[k].lower() == "as":
                        if (k + 1) < len(tokens) and _is_identifier(tokens[k + 1]):
                            alias_to_base[tokens[k + 1].lower()] = base
                            i = k + 2
                            continue
                    elif (
                        _is_identifier(tokens[k])
                        and tokens[k].lower() not in SQL_KEYWORDS
                    ):
                        alias_to_base[tokens[k].lower()] = base
                        i = k + 1
                        continue
                i = jnext
                continue
            out.append(tok)
            i += 1
            continue

        if _is_identifier(tok):
            parts, jnext = _read_dotted_identifier(tokens, i)
            if len(parts) >= 2:
                base = alias_to_base.get(parts[-2], parts[-2])
                out.extend([table_id(base), ".", column_id(parts[-1])])
                i = jnext
                continue
            name = parts[0] if parts else tokens[i].lower()
            if name in table_map or name in alias_to_base:
                out.append(table_id(alias_to_base.get(name, name)))
            elif (i + 1) < len(tokens) and tokens[i + 1] == "(":
                out.append(name)
            else:
                out.append(column_id(name))
            i = jnext if parts else (i + 1)
            continue

        out.append(tok)
        i += 1

    spaced = []
    for j, tok in enumerate(out):
        if (
            j > 0
            and tok not in {".", ",", ")", ";"}
            and out[j - 1] not in {"(", ".", ","}
        ):
            spaced.append(" ")
        spaced.append(tok)
    return "".join(spaced)


class SqlAstEncoder:
    BASE_KEYS = [
        "tables",
        "joins",
        "left",
        "right",
        "inner",
        "outer",
        "where",
        "group",
        "order",
        "having",
        "subq",
        "distinct",
        "union",
        "limit",
        "preds",
        "proj",
    ]
    BASE_SCALES = [8, 4, 4, 4, 4, 4, 1, 1, 1, 1, 4, 1, 1, 1, 10, 8]

    def __init__(self, struct_dim: int = 16):
        self.struct_dim = struct_dim

    def compute_signature(self, sql: str) -> Dict[str, int]:
        try:
            ast = sqlglot.parse_one(sql)
            tables = {t.name.lower() for t in ast.find_all(sqlglot.exp.Table) if t.name}
            joins = [(j.kind or "join").lower() for j in ast.find_all(sqlglot.exp.Join)]
            sel = ast.find(sqlglot.exp.Select)
            proj = len(sel.expressions) if sel else 0
            return dict(
                tables=len(tables),
                joins=len(joins),
                left=joins.count("left"),
                right=joins.count("right"),
                inner=joins.count("inner"),
                outer=joins.count("outer"),
                where=int(ast.find(sqlglot.exp.Where) is not None),
                group=int(ast.find(sqlglot.exp.Group) is not None),
                order=int(ast.find(sqlglot.exp.Order) is not None),
                having=int(ast.find(sqlglot.exp.Having) is not None),
                subq=len(list(ast.find_all(sqlglot.exp.Subquery))),
                distinct=int(
                    any(
                        isinstance(s, sqlglot.exp.Select) and s.args.get("distinct")
                        for s in ast.find_all(sqlglot.exp.Select)
                    )
                ),
                union=int(ast.find(sqlglot.exp.Union) is not None),
                limit=int(ast.find(sqlglot.exp.Limit) is not None),
                preds=sum(1 for _ in ast.find_all(sqlglot.exp.EQ))
                + sum(
                    1
                    for _ in ast.find_all(
                        sqlglot.exp.GT,
                        sqlglot.exp.GTE,
                        sqlglot.exp.LT,
                        sqlglot.exp.LTE,
                        sqlglot.exp.NEQ,
                    )
                ),
                proj=min(proj, 8),
            )
        except Exception:
            s = sql.lower()
            return dict(
                tables=len(re.findall(r"\bfrom\b|\bjoin\b", s)),
                joins=len(re.findall(r"\bjoin\b", s)),
                where=int(" where " in s),
                group=int(" group by " in s),
                order=int(" order by " in s),
                having=int(" having " in s),
                subq=s.count("(select "),
                distinct=int(" distinct " in s),
                union=int(" union " in s),
                limit=int(" limit " in s),
                preds=len(re.findall(r"=|<>|!=|>=|<=|>|<", s)),
                proj=1,
            )

    def structure_vector(self, sql: str) -> np.ndarray:
        sig = self.compute_signature(sql)
        v = np.array(
            [
                sig.get(k, 0) / max(self.BASE_SCALES[i], 1)
                for i, k in enumerate(self.BASE_KEYS)
            ],
            dtype=np.float32,
        )
        v /= np.clip(np.linalg.norm(v), 1e-12, None)
        if len(v) < self.struct_dim:
            v = np.pad(v, (0, self.struct_dim - len(v)))
        else:
            v = v[: self.struct_dim]
        return v.astype(np.float32)

    def encode(self, sql_queries: List[str]) -> np.ndarray:
        matrix = np.stack([self.structure_vector(sql) for sql in sql_queries]).astype(
            np.float32
        )
        return matrix / np.clip(
            np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None
        )


TEXT_DIM = 384
AST_DIM = 16
DIMENSIONS = {
    "text": TEXT_DIM,
    "annoy": TEXT_DIM,
    "text_annoy": 2 * TEXT_DIM,
    "text_ast": TEXT_DIM + AST_DIM,
}


def create_query_encoder(name, text_model):
    ast_encoder = SqlAstEncoder()

    def encode_text(sql):
        return text_model.encode(sql).tolist()

    def encode_anonymized(sql):
        return (
            text_model.encode([anonymize_sql(sql)], convert_to_numpy=True)[0]
            .astype(np.float32)
            .tolist()
        )

    def encode_text_and_anonymized(sql):
        emb_orig = text_model.encode([sql], convert_to_numpy=True)[0]
        emb_anon = text_model.encode([anonymize_sql(sql)], convert_to_numpy=True)[0]
        return np.concatenate([emb_orig, emb_anon]).astype(np.float32).tolist()

    def encode_text_and_ast(sql):
        text_emb = text_model.encode([sql], convert_to_numpy=True)[0]
        return (
            np.concatenate([text_emb, ast_encoder.encode([sql])[0]])
            .astype(np.float32)
            .tolist()
        )

    encoders = {
        "text": encode_text,
        "annoy": encode_anonymized,
        "text_annoy": encode_text_and_anonymized,
        "text_ast": encode_text_and_ast,
    }
    if name not in encoders:
        raise ValueError(f"Unknown encoder: {name}")
    return encoders[name], DIMENSIONS[name]
