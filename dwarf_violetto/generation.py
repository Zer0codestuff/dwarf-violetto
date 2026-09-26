"""Generation ends at EOS unless a caller requests a smaller output budget."""


def output_budget(prompt_length, context_length, requested=None):
    remaining = context_length - prompt_length
    if remaining < 1:
        raise ValueError("The prompt leaves no room in the model context")
    if requested is not None and requested < 1:
        raise ValueError("An explicit output budget must be positive")
    return remaining if requested is None else min(requested, remaining)


def finish_reason(reason, prompt_length, context_length, requested=None):
    if reason != "length":
        return reason
    if requested is None or requested >= context_length - prompt_length:
        return "context_limit"
    return "length"
