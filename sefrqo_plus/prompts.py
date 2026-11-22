FORMAT_RULES = (
    "NO OTHER REDUNDANT OUTPUTS OR COMMENTS. IMPORTANT: The hint must include an extra outer pair of parentheses in the Leading clause, and the brackets should match. "
    "For example, given a join order of 'a b c d':\n"
    "Valid hint: /*+ Leading((((a b) c) d)) */\n"
    "Invalid hint (missing extra outer parentheses): /*+ Leading(((a b) c) d) */\n"
    "Invalid hint (brackets do not match): /*+ Leading(((((a b) c) d)) */\n"
    "Invalid hint (brackets do not match): /*+ Leading(((a b) c) d)) */\n"
)


def build_references_section(num_references, reference_stats, reference_plans):
    text = (
        f"## Here is {num_references} similar query-answer pairs you can reference:\n"
    )
    for j in range(num_references):
        text += reference_stats[j] + "\nThe optimal hint:\n" + reference_plans[j] + "\n"
    return text


def build_first_prompt(question_stats, postgres_hint, compact_instruction=False):
    return (
        f"## Here is the query and its corresponding statistics: \n{question_stats}\n"
        f"The hint provided by Postgres: \n{postgres_hint}\n"
        "## Please provide a better hint in the same format as Postgres's. Return ONLY the final SQL hint"
        + ("" if compact_instruction else ". ")
        + "DO NOT USE THE SAME HINT AS THE Postgres HINT.\n"
        + FORMAT_RULES
    )


def build_feedback_prompt(
    question_stats, postgres_hint, best_hint, gain_str, with_references
):
    request = "a different hint" if with_references else "a hint"
    return (
        f"## Here is the query and its corresponding statistics: \n{question_stats}\n"
        f"The hint provided by Postgres: \n{postgres_hint}\n"
        f"The current best hint provided by LLM: \n{best_hint}\n"
        f"The current performance gain is {gain_str}\n"
        "If the performance gain is negative, it means that the current LLM-generated plan is inferior to the Postgres plan.\n"
        f"## Please generate {request} that at least meets or exceeds their best performance in the same format. Return ONLY the final SQL hint."
        "DO NOT USE THE SAME HINT AS THE CURRENT BEST HINT or Postgres HINT.\n"
        + FORMAT_RULES
    )


def format_gain(postgres_time, best_time):
    if postgres_time == 0 or best_time == 0:
        return "UNKNOWN"
    return f"{(postgres_time - best_time) / postgres_time * 100:.3f}%"


FIRST_GENERATION_NOTE = "No improvement over the first generation so far. Please do your best to generate a better query hint.\n"


def build_plus_references_section(
    num_references, reference_stats, reference_plans, verb="is", clamp=False
):
    k = (
        min(num_references, len(reference_plans), len(reference_stats))
        if clamp
        else num_references
    )
    text = f"## Here {verb} {k} similar query-answer pairs you can reference:\n"
    for j in range(k):
        text += reference_stats[j] + "\nThe optimal hint:\n" + reference_plans[j] + "\n"
    return text


def build_plus_first_prompt(question_stats):
    return (
        f"## Here is the query and its corresponding statistics: \n{question_stats}\n"
        "## Please provide a query hint in the correct format. Return ONLY the final SQL hint. "
        + FORMAT_RULES
    )


def is_valid_time(t):
    return isinstance(t, (int, float)) and t > 0


def format_best_gain_line(first_time, best_time):
    if is_valid_time(first_time) and is_valid_time(best_time):
        if first_time == best_time:
            return FIRST_GENERATION_NOTE
        return f"The current best performance gain (vs. your first generation) is {(first_time - best_time) / first_time * 100.0:.3f}%\n"
    return "Baseline or best record is unavailable.\n"


def format_recent_gain_line(first_time, recent_time):
    if is_valid_time(first_time) and is_valid_time(recent_time):
        return f"The most recent performance gain (vs. your first generation) is {(first_time - recent_time) / first_time * 100.0:.3f}%\n"
    return "Baseline or most recent record is unavailable.\n"


def build_plus_feedback_prompt(
    question_stats, best_hint, best_line, recent_hint, recent_line
):
    return (
        f"## Here is the query and its corresponding statistics:\n{question_stats}\n"
        f"The current best hint provided by you:\n{best_hint}\n"
        f"{best_line}"
        f"The most recent hint provided by you:\n{recent_hint}\n"
        f"{recent_line}"
        "If the performance gain is negative, it means that the plan is inferior to the first-generation plan.\n"
        "## Please generate a different hint that at least meets or exceeds the best performance shown above, in the same format."
        "Return ONLY the final SQL hint.\n"
        "DO NOT USE THE SAME HINT AS THE CURRENT BEST HINT or the MOST RECENT HINT.\n"
        + FORMAT_RULES.replace(
            "should match. For example", "should match.\nFor example"
        )
    )
