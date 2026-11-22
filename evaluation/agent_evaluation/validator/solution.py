"""The validator you write. This is the only file you need to modify.

Each function evaluates a single run object and returns the errors it finds.
An empty list means that no errors were found. `validator/runner.py` defines
what a run object contains, and prediction.py defines the four error families.
"""

import base64
import io
import warnings

from openai import BadRequestError
from PIL import Image, UnidentifiedImageError

from agent_evaluation.validator.model import complete, detokenize, tokenize
from agent_evaluation.validator.prediction import Error, ErrorFamily
from agent_evaluation.validator.runner import Run


CONTEXT_FRACTION = 0.5
MAX_OUTPUT_TOKENS = 250
OMISSION = "\n[... omitted to fit model context ...]\n"
COMPACTION_NOTE = "\nSome evidence may be omitted. Do not treat omissions as agent errors.\n"
#: Shrink the image's longest side by this factor each time text compaction alone
#: still overflows. The server's tokenizer does not count image tokens, so we let
#: its context errors drive downscaling instead of estimating a pixel budget.
IMAGE_DOWNSCALE_FACTOR = 0.7
#: Stop shrinking once the longest side reaches this floor; below it the figure is
#: unreadable and further retries cannot help, so we surface the overflow instead.
MIN_IMAGE_LONG_SIDE = 64


def _excerpt(text: str, tokens: list[int], limit: int) -> str:
    """Retain the beginning and end, including the run's final actions."""
    if len(tokens) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return (
        (detokenize(tokens[:head]) if head else "")
        + OMISSION
        + (detokenize(tokens[-tail:]) if tail else "")
    )


def _compact(trajectory: str, inputs: str, fixed_prompt: str) -> tuple[str, str]:
    trajectory_info = tokenize(trajectory)
    context = trajectory_info.get("max_model_len")
    if type(context) is not int or context <= 0:
        raise RuntimeError("Validator server did not report a valid max_model_len")
    trajectory_tokens = trajectory_info["tokens"]
    input_tokens = tokenize(inputs)["tokens"]
    # Reserve half the context for the image, chat template, and tokenization
    # boundary differences. Count fixed text and omission markers separately.
    fixed_tokens = tokenize(fixed_prompt + COMPACTION_NOTE + OMISSION * 2)["tokens"]
    budget = int(context * CONTEXT_FRACTION) - len(fixed_tokens) - MAX_OUTPUT_TOKENS
    if budget < 0:
        raise RuntimeError("Task instructions exceed the baseline's context budget")
    total = len(trajectory_tokens) + len(input_tokens)
    trajectory_budget = budget * len(trajectory_tokens) // max(total, 1)
    return (
        _excerpt(trajectory, trajectory_tokens, trajectory_budget),
        _excerpt(inputs, input_tokens, budget - trajectory_budget),
    )


def _downscale(data: bytes) -> bytes | None:
    """Shrink a PNG's longest side by IMAGE_DOWNSCALE_FACTOR.

    Returns the smaller PNG, or None when the image cannot be shrunk further
    (already at the floor) or cannot be decoded at all.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            width, height = image.size
            mode = image.mode
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    longest = max(width, height)
    if longest <= MIN_IMAGE_LONG_SIDE:
        return None
    scale = max(MIN_IMAGE_LONG_SIDE / longest, IMAGE_DOWNSCALE_FACTOR)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if new_size == (width, height):
        return None
    with Image.open(io.BytesIO(data)) as image:
        if mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")
        resized = image.resize(new_size, Image.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    return buffer.getvalue()


def _context_overflow(error: BadRequestError) -> bool:
    message = str(error).lower()
    return any(phrase in message for phrase in (
        "context_length_exceeded",
        "maximum context length",
        "maximum model length",
        "maximum input length",
    ))


def _ask(run: Run, family: ErrorFamily, question: str) -> list[Error]:
    inputs = "\n".join(
        f"{name}:\n{path.read_text(errors='replace')}" for name, path in run.inputs.items()
    )
    trajectory = str(run.messages)
    image_bytes = run.figure.read_bytes() if run.figure else None

    def make_prompt(shown_trajectory: str, shown_inputs: str) -> str:
        return f"""{question}

Request: {run.instructions}
Trajectory: {shown_trajectory}
Input files: {shown_inputs or '(none)'}
"""

    def ask(prompt: str, image: bytes | None) -> dict:
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            encoded = base64.b64encode(image).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            )
        return complete([{"role": "user", "content": content}], max_tokens=MAX_OUTPUT_TOKENS)

    try:
        response = ask(make_prompt(trajectory, inputs), image_bytes)
    except BadRequestError as error:
        if not _context_overflow(error):
            raise
        warnings.warn(
            f"Run {run.run_id} ({family.value}): model context exceeded; "
            f"retrying with text capped at {CONTEXT_FRACTION:.0%} of server context.",
            stacklevel=2,
        )
        shown_trajectory, shown_inputs = _compact(trajectory, inputs, make_prompt("", ""))
        prompt = make_prompt(shown_trajectory, shown_inputs) + COMPACTION_NOTE
        # Compacted text still overflows only when the image dominates the context.
        # Downscale it in place, retrying until the server accepts it or the figure
        # reaches a floor below which it would be unreadable anyway.
        image = image_bytes
        while True:
            try:
                response = ask(prompt, image)
                break
            except BadRequestError as retry_error:
                if not _context_overflow(retry_error):
                    raise
                smaller = _downscale(image) if image is not None else None
                if smaller is None:
                    raise
                with Image.open(io.BytesIO(smaller)) as shrunk:
                    new_size = shrunk.size
                warnings.warn(
                    f"Run {run.run_id} ({family.value}): still over context; "
                    f"downscaling figure to {new_size[0]}x{new_size[1]} and retrying.",
                    stacklevel=2,
                )
                image = smaller
    answer = response["choices"][0]["message"]["content"].strip()
    if not "<YES>" in answer.upper():
        return []
    return [Error(family=family, evidence=answer[3:].lstrip(" :.-\n") or answer)]


def judge_execution(run: Run) -> list[Error]:
    """Whether the run produced a figure to judge at all."""
    if run.figure:
        return []

    return [Error(family=ErrorFamily.EXECUTION_FAILURE, evidence="No output figure detected")]


def judge_data_and_chart(run: Run) -> list[Error]:
    """Whether the figure plots the requested data, built the requested way."""
    wrong_data = _ask(
        run,
        ErrorFamily.WRONG_DATA,
        """Does the plotted data differ from what was requested? Below is the evaluation criteria. A criteria (e.g. order) that is not related with the request can be skipped.
        1. Wrong values. Plotted numbers, positions, or derived quantities differ from the requested data or formula: wrong aggregation, scaling, or normalisation; a function or field sampled over the wrong range or grid; a bubble size, colour value, or error bar that should encode the data but does not.
        2. Missing data. A requested series, category, group, panel's data, or data point is not drawn, or a dimension of the data is dropped (a 3D quantity plotted with one variable missing).
        3. Wrong selection. The wrong subset, column, or filter was used, or series that were not requested are mixed in with the requested ones.
        4. Wrong order. Points, bars, categories, or sequences appear in a different order from the one requested: unsorted when sorting was asked for, a sequence reversed or shuffled, panels that do not show the data they were assigned.
        
        After your analysis, if it deffers, start your answer with <YES>, otherwise start with <NO>. After that briefly explain the reason. """,
    )
    wrong_chart = _ask(
        run,
        ErrorFamily.WRONG_CHART,
        """Does the figure fail to strictly follow any part of the requested chart design? Below is the evaluation criteria. Evaluate strictly against the stated requirements. Treat any requirement that is only partially met as a failure. A criteria that is not related with the request can be skipped. You should answer the question based on the output image, not the trajectory!
1. Chart type and projection. Wrong chart type (a line for a bar chart, no pie), grouped bars for stacked, vertical bars for horizontal, 3D requested but drawn in 2D, polar projection missing, contour or surface missing, twin axes missing or used when forbidden, a zoomed inset or its connector lines missing.
Layout. 
2. Wrong number or arrangement of subplots, wrong figure size or DPI, panels that do not share the axes they were asked to share.
Axes. 
3. Missing or wrong x, y, or z axis label; wrong exact title or label text; wrong axis limits (not starting from zero, wrong range, limits that cut off requested data); wrong axis scale (linear for log, non-log contour levels); ticks or tick labels shown when asked to be hidden; wrong tick labels or tick-label rotation; spines visible when asked to be hidden; grid missing.
4. Legend, colour bar, and annotations. Legend missing; colour bar missing or not matching the data; a requested annotation or marked point missing or placed at the wrong location; a requested function label or bar value label missing; a requested formula not rendered; the wrong region shaded.
5. Style. Wrong or forbidden colormap (a non-uniform one when a perceptually uniform one was requested); wrong line style (solid for dashed or dotted); wrong marker shape; wrong colour for a named series; areas not filled; whiskers, median markers, connector lines, or boxes missing; polygons not closed; requested line offsets not applied.

After your analysis, start your answer with <YES> if you identify a failure, or <NO> otherwise. Then briefly explain your reasoning, citing specific evidence."""
    )
    return wrong_data + wrong_chart


def judge_readability(run: Run) -> list[Error]:
    return _ask(
        run,
        ErrorFamily.HARD_TO_READ,
        """Is the rendered figure difficult or impossible to read? Below is the evaluation critiera. 
        1. Clipped text. A title, axis label, tick label, or annotation is partly cut off by the edge of the image (a label cut in half is hard_to_read; one that is entirely outside the image is wrong_chart, see above).
        2. Overlapping text. Tick labels, annotations, titles, or labels overlap one another, or markers are drawn on top of the text that labels them.
        3. Elements covering content. A legend, annotation, or text box hides data, labels, or other text. A legend sitting over empty space is fine.
        4. Low contrast or invisible content. Text or value labels with a contrast ratio below about 2.5:1 against their background; a series drawn in the background colour, with zero width, or otherwise invisible.
        5. Indistinguishable data. Two series or groups that must be told apart are drawn with the same, or nearly the same, colour, line style, and marker; a colormap that merges values that should look different.
        6. Squeezed layout. Data, boxes, or panels crushed into an unreadable sliver by axis limits or by axes that overlap each other.
        After your analysis, if the figure is not readable, start your answer with <YES>, otherwise start with <NO>.  Then briefly explain your reasoning. """,
    )


def validate(run: Run) -> list[Error]:
    execution = judge_execution(run)
    if execution:
        return execution
    return judge_data_and_chart(run) + judge_readability(run)
