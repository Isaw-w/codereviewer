import argparse
import json
import os
import queue
import random
import threading
import time

from tqdm import tqdm

from prompts.create_prompts_for_rubric_eval import \
    create_prompt_template_judge_model
from utils import (get_judge_response, load_existing_indices_as_set,
                   prepare_criterion_data, setup_client)

parser = argparse.ArgumentParser(description='Run judge model on generated responses (MoReBench)')
parser.add_argument("--input_file", "-i", required=True, help="Path to input JSONL file with generations")
parser.add_argument("--api_key", "-ak", required=True, help="API key for OpenRouter")
parser.add_argument("--judgement_type", "-jt", default="thinking_trace", 
                    choices=["model_resp", "thinking_trace"], 
                    help="Which field to judge: model_resp or thinking_trace")
parser.add_argument("--num_parallel_request", "-n", type=int, default=100, 
                    help="Number of parallel requests")
parser.add_argument("--debug", "-d", action='store_true', help='Debug with only 5 examples')
parser.add_argument("--judge_model", "-jm", default="openai/gpt-oss-120b", 
                    help="Judge model to use (default: openai/gpt-oss-120b)")
parser.add_argument("--temperature", type=float, default=1.0,
                    help="Judge temperature (default: 1.0)")
parser.add_argument("--top_p", type=float, default=1.0,
                    help="Judge top_p (default: 1.0)")
parser.add_argument("--max_tokens", type=int, default=10500,
                    help="Judge max_tokens (default: 10500)")
parser.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="medium",
                    help="Judge reasoning effort (default: medium)")
parser.add_argument("--expected_samples", "-es", type=int, default=500,
                    help="Expected number of samples in input file")
parser.add_argument("--output_dir", "-o", default=None,
                    help="Output directory (default: auto-generated from input path)")
parser.add_argument("--stall_timeout_sec", type=float,
                    default=float(os.getenv("JUDGE_STALL_TIMEOUT_SEC", "60")),
                    help="Abort after this many seconds with no completed judgements; unfinished idx will be logged for resume repair")
parser.add_argument("--stall_poll_sec", type=float, default=5.0,
                    help="Polling interval for no-progress stall detection")
parser.add_argument("--request_timeout_sec", type=float,
                    default=float(os.getenv("JUDGE_REQUEST_TIMEOUT_SEC", "60")),
                    help="Per-request deadline in seconds; overdue requests are skipped to .errors.jsonl and resume repair handles them later")

args = parser.parse_args()

# Setup output filename
if args.output_dir:
    output_filename = os.path.join(args.output_dir, 
                                   f"{args.judgement_type}_{os.path.basename(args.input_file)}")
else:
    output_filename = args.input_file.replace("generations/", 
                                              f"{args.judgement_type}_judgements/")

assert args.input_file != output_filename, "Input and output filenames must be different"
error_filename = output_filename + ".errors.jsonl"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

print(f"Input file: {args.input_file}")
print(f"Output file: {output_filename}")
print(f"Error file: {error_filename}")
print(f"Judge model: {args.judge_model}")
print(f"Judging: {args.judgement_type}")
print(f"Per-request timeout: {args.request_timeout_sec}s")

# Load data
with open(args.input_file, "r") as f:
    data = [json.loads(line) for line in f.readlines()]


if args.debug:
    data = data[:5]
    print("DEBUG MODE: Processing only 5 samples")
else:
    assert len(data) == args.expected_samples, f"Expected {args.expected_samples} samples but found {len(data)}"


# Prepare criterion data
criterion_data = prepare_criterion_data(data, args.judgement_type)

# Shuffle to increase cache hit
random.seed(42)
random.shuffle(criterion_data)

# Setup client
client = setup_client('openrouter', args.api_key)


def get_judgement(idx, dp):
    """Get judgment for a single criterion"""
    reasoning_resp = dp["response"]
    rubric_criterion = dp["criterion"]
    instruction_prompt = create_prompt_template_judge_model()
    prompt = f'Reasoning Response:{reasoning_resp}\n\n{instruction_prompt}\n\nRubric Criterion:{rubric_criterion}'

    response = ""
    input_tokens = 0
    output_tokens = 0
    empty_retry_used = False
    for attempt in range(2):
        response, input_tokens, output_tokens = get_judge_response(
            client,
            args.judge_model,
            prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        if isinstance(response, str) and response.strip():
            break
        empty_retry_used = True
        response = ""

    if not isinstance(response, str) or not response.strip():
        raise RuntimeError("Empty judge response after retry")

    dp['idx'] = idx
    dp["judgement"] = response if isinstance(response, str) else ""
    dp["judge_input_tokens"] = input_tokens
    dp["judge_output_tokens"] = output_tokens
    if empty_retry_used:
        dp["judge_empty_retry_used"] = True
    return dp


def make_error_row(idx, dp, reason, *, timeout_skipped=False):
    row = dict(dp)
    row["idx"] = idx
    row["judgement"] = ""
    row["judge_input_tokens"] = 0
    row["judge_output_tokens"] = 0
    row["judge_error"] = reason
    row["judge_timeout_skipped"] = timeout_skipped
    row["judge_error_ts"] = int(time.time())
    return row

# Process judgments
with open(output_filename, "a+", encoding="utf-8") as fw, open(error_filename, "a+", encoding="utf-8") as ew:
    fw.seek(0)
    existing_idx = load_existing_indices_as_set(output_filename)
    print(f"Found {len(existing_idx)} existing rows out of {len(criterion_data)}")

    pending_items = [
        (idx, dp)
        for idx, dp in enumerate(criterion_data)
        if idx not in existing_idx
    ]
    print(f"Submitting {len(pending_items)} pending judgements with stall_timeout_sec={args.stall_timeout_sec}")

    if pending_items:
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        pending_by_idx = {idx: dp for idx, dp in pending_items}
        in_flight = {}
        state_lock = threading.Lock()
        timed_out_idx = set()
        run_state = {
            "error_count": 0,
            "last_progress_ts": time.time(),
        }

        for item in pending_items:
            task_queue.put(item)

        def worker():
            while True:
                try:
                    idx, dp = task_queue.get_nowait()
                except queue.Empty:
                    return
                with state_lock:
                    in_flight[idx] = time.time()
                try:
                    result = get_judgement(idx, dp)
                    result_queue.put(("ok", idx, result))
                except Exception as exc:
                    result_queue.put(("err", idx, str(exc)))
                finally:
                    task_queue.task_done()

        worker_count = min(args.num_parallel_request, len(pending_items))
        max_worker_budget = min(len(pending_items), max(worker_count, worker_count * 3))
        workers = []

        def start_worker():
            t = threading.Thread(target=worker, name=f"judge-worker-{len(workers)}", daemon=True)
            t.start()
            workers.append(t)

        for _ in range(worker_count):
            start_worker()

        progress = tqdm(total=len(pending_items))
        last_stall_report_ts = 0.0
        poll_timeout = min(args.stall_poll_sec, 1.0)

        def expire_slow_requests(now):
            expired = []
            with state_lock:
                for idx, start_ts in list(in_flight.items()):
                    if idx in timed_out_idx:
                        continue
                    if now - start_ts >= args.request_timeout_sec:
                        expired.append(idx)
                        timed_out_idx.add(idx)
                        in_flight.pop(idx, None)
            for idx in expired:
                dp = pending_by_idx.pop(idx, None)
                if dp is None:
                    continue
                ew.write(
                    json.dumps(
                        make_error_row(
                            idx,
                            dp,
                            f"request_timeout_{args.request_timeout_sec:.1f}s",
                            timeout_skipped=True,
                        )
                    )
                    + "\n"
                )
                ew.flush()
                run_state["error_count"] += 1
                progress.update(1)
                run_state["last_progress_ts"] = now
                print(
                    f"[request-timeout] idx={idx} exceeded {args.request_timeout_sec:.1f}s; "
                    f"skipped for later resume repair"
                )
                if not task_queue.empty() and len(workers) < max_worker_budget:
                    start_worker()

        while pending_by_idx:
            expire_slow_requests(time.time())
            try:
                status, idx, payload = result_queue.get(timeout=poll_timeout)
            except queue.Empty:
                expire_slow_requests(time.time())
                idle_sec = time.time() - run_state["last_progress_ts"]
                now = time.time()
                if now - last_stall_report_ts >= args.stall_poll_sec:
                    print(
                        f"[stall-watch] no completed judgements for {idle_sec:.1f}s; "
                        f"pending={len(pending_by_idx)}"
                    )
                    last_stall_report_ts = now
                if idle_sec >= args.stall_timeout_sec:
                    for stuck_idx, stuck_dp in sorted(pending_by_idx.items()):
                        ew.write(
                            json.dumps(
                                make_error_row(
                                    stuck_idx,
                                    stuck_dp,
                                    "stuck_timeout_no_progress",
                                    timeout_skipped=True,
                                )
                            )
                            + "\n"
                        )
                    ew.flush()
                    progress.close()
                    raise SystemExit(
                        f"Stall timeout reached after {idle_sec:.1f}s with {len(pending_by_idx)} unfinished judgements. "
                        f"Wrote timeout skips to {error_filename} for later resume repair."
                    )
                continue

            with state_lock:
                in_flight.pop(idx, None)
            dp = pending_by_idx.pop(idx, None)
            if dp is None:
                continue

            if status == "ok":
                fw.write(json.dumps(payload) + "\n")
                fw.flush()
            else:
                ew.write(json.dumps(make_error_row(idx, dp, payload, timeout_skipped=False)) + "\n")
                ew.flush()
                run_state["error_count"] += 1

            progress.update(1)
            run_state["last_progress_ts"] = time.time()

        progress.close()
        if run_state["error_count"]:
            raise SystemExit(
                f"Completed with {run_state['error_count']} skipped/failed judgements. "
                f"See {error_filename}; rerun to resume repair."
            )

print(f"Completed! Results written to: {output_filename}")
