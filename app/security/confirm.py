"""Interactive confirmation for risky actions (stdin, no silent auto-approve)."""

from __future__ import annotations


def confirm_risky_action(*, tool_name: str, summary: str) -> bool:
    """
    Block until the user explicitly approves or denies.
    Returns True only for a clear affirmative.
    """
    prompt = (
        f"\n[SECURITY] The tool '{tool_name}' is marked high-risk.\n"
        f"Summary: {summary}\n"
        "Type 'yes' to proceed, anything else to cancel: "
    )
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer == "yes"
