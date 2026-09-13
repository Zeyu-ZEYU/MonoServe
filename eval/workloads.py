"""Request sets drawn from ShareGPT and LongBench v2.

ShareGPT: the first human turn of every conversation that opens with a
human turn followed by an assistant turn (the convention of vLLM's serving
benchmark). LongBench v2: the zero-shot multiple-choice prompt of the
LongBench v2 evaluation, built from the context, the question and the four
choices. Both files come from the Hugging Face Hub and stay in its local
cache.

Candidates are shuffled with the seed and kept in that order while their
prompts fit --max-prompt-tokens, up to --num requests. Every request asks
for max_model_len - prompt_tokens output tokens, so its output runs to the
end of the sequence (the requests never set ignore_eos). Prompts are sent as
raw text to /v1/completions; --chat-template wraps each one in the
tokenizer's chat template as a single user turn instead.

    python -m eval.workloads --dataset sharegpt --tokenizer <model> \
        --max-model-len 131072 --num 2000 --model qwen3-30b

Output: <results>/<model>/<dataset>/workload.jsonl (or --out). The header
line holds the settings and the reference request, a prompt of 1024 tokens
whose solo TTFT is the floor of every TTFT target (see solo.py); each
further line is one request: {"id", "prompt", "prompt_tokens",
"max_tokens"}. prompt_tokens counts the tokens the model's tokenizer
produces for the prompt with its default special tokens, as vLLM's
completions endpoint tokenizes it.
"""
import argparse
import json
import random

from .common import DATASETS, log, read_jsonl, workload_path, write_jsonl

SOURCES = {
    "sharegpt": ("anon8231489123/ShareGPT_Vicuna_unfiltered",
                 "ShareGPT_V3_unfiltered_cleaned_split.json"),
    "longbench": ("THUDM/LongBench-v2", "data.json"),
}

LONGBENCH_TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""

REFERENCE_ID = "reference-1k"
REFERENCE_OUTPUT = 16   # only the reference's TTFT is used


def download(dataset, cache_dir=None):
    from huggingface_hub import hf_hub_download
    repo, name = SOURCES[dataset]
    return hf_hub_download(repo, name, repo_type="dataset", cache_dir=cache_dir)


def sharegpt_prompts(path):
    """(id, first human turn) per usable conversation, in file order."""
    out = []
    with open(path) as f:
        data = json.load(f)
    for i, c in enumerate(data):
        conv = c.get("conversations") or []
        if (len(conv) < 2 or conv[0].get("from") != "human"
                or conv[1].get("from") != "gpt"):
            continue
        text = conv[0].get("value") or ""
        if text.strip():
            out.append((f"sharegpt-{i}", text))
    return out


def longbench_prompt(item):
    # The same chain of replacements as the LongBench v2 evaluation code.
    return (LONGBENCH_TEMPLATE.replace("$DOC$", item["context"].strip())
            .replace("$Q$", item["question"].strip())
            .replace("$C_A$", item["choice_A"].strip())
            .replace("$C_B$", item["choice_B"].strip())
            .replace("$C_C$", item["choice_C"].strip())
            .replace("$C_D$", item["choice_D"].strip()))


def longbench_prompts(path):
    with open(path) as f:
        return [(it["_id"], longbench_prompt(it)) for it in json.load(f)]


READERS = {"sharegpt": sharegpt_prompts, "longbench": longbench_prompts}


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text))


def as_user_turn(tokenizer, text):
    return tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                         tokenize=False,
                                         add_generation_prompt=True)


def build(candidates, tokenizer, max_model_len, *, max_prompt_tokens=None,
          max_output_tokens=None, num=None, seed=0, chat_template=False):
    """Shuffle the (id, text) candidates with the seed and keep, in that
    order, up to num requests whose prompts have at most max_prompt_tokens
    tokens (default: max_model_len - 1024)."""
    limit = max_prompt_tokens or max_model_len - 1024
    if limit < 1:
        raise ValueError("no prompt fits; set max_prompt_tokens")
    order = list(candidates)
    random.Random(seed).shuffle(order)
    out = []
    for rid, text in order:
        prompt = as_user_turn(tokenizer, text) if chat_template else text
        n = count_tokens(tokenizer, prompt)
        if n > limit or n >= max_model_len:
            continue
        m = max_model_len - n
        out.append({"id": rid, "prompt": prompt, "prompt_tokens": n,
                    "max_tokens": min(m, max_output_tokens or m)})
        if num and len(out) >= num:
            break
    return out


def reference_request(texts, tokenizer, tokens=1024, chat_template=False):
    """A prompt of about `tokens` tokens: the leading tokens of the given
    texts joined by blank lines, decoded back to text."""
    body = tokens
    if chat_template:
        body -= count_tokens(tokenizer, as_user_turn(tokenizer, ""))
    joined, ids = "", []
    for t in texts:
        t = t[:16 * body]     # enough characters for body tokens of text
        joined = f"{joined}\n\n{t}" if joined else t
        ids = tokenizer.encode(joined, add_special_tokens=False)
        if len(ids) >= body:
            break
    if len(ids) < body:
        raise ValueError(f"the texts hold only {len(ids)} tokens")
    prompt = tokenizer.decode(ids[:body], skip_special_tokens=True)
    if chat_template:
        prompt = as_user_turn(tokenizer, prompt)
    return {"id": REFERENCE_ID, "prompt": prompt,
            "prompt_tokens": count_tokens(tokenizer, prompt),
            "max_tokens": REFERENCE_OUTPUT}


def load_workload(path):
    """(header, requests) of a workload file."""
    return read_jsonl(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--tokenizer", required=True,
                    help="model path or Hugging Face id of the served model")
    ap.add_argument("--max-model-len", type=int, required=True,
                    help="context limit of the servers (prompt + output)")
    ap.add_argument("--max-prompt-tokens", type=int,
                    help="drop longer prompts (default: max-model-len - 1024)")
    ap.add_argument("--max-output-tokens", type=int,
                    help="cap max_tokens, e.g. for a quick trial run; by "
                         "default outputs are limited only by the context")
    ap.add_argument("--num", type=int, default=2000,
                    help="at most this many requests (0: all that fit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat-template", action="store_true")
    ap.add_argument("--reference-tokens", type=int, default=1024)
    ap.add_argument("--cache-dir", help="Hugging Face cache directory")
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", help="model name in the results layout")
    ap.add_argument("--out", help="output file (default: the results layout)")
    args = ap.parse_args()
    if not args.out and not args.model:
        ap.error("give --model or --out")
    out = args.out or workload_path(args.results, args.model, args.dataset)

    path = download(args.dataset, args.cache_dir)
    candidates = READERS[args.dataset](path)
    log(f"{args.dataset}: {len(candidates)} candidate prompts")
    tok = load_tokenizer(args.tokenizer)
    reqs = build(candidates, tok, args.max_model_len,
                 max_prompt_tokens=args.max_prompt_tokens,
                 max_output_tokens=args.max_output_tokens,
                 num=args.num or None, seed=args.seed,
                 chat_template=args.chat_template)
    ref = reference_request([t for _, t in candidates], tok,
                            args.reference_tokens, args.chat_template)
    header = {"dataset": args.dataset, "tokenizer": args.tokenizer,
              "seed": args.seed, "max_model_len": args.max_model_len,
              "max_prompt_tokens": (args.max_prompt_tokens
                                    or args.max_model_len - 1024),
              "max_output_tokens": args.max_output_tokens,
              "chat_template": args.chat_template, "num": len(reqs),
              "reference": ref}
    write_jsonl(out, header, reqs)
    lens = sorted(r["prompt_tokens"] for r in reqs)
    log(f"wrote {out}: {len(reqs)} requests, prompt tokens median "
        f"{lens[len(lens) // 2] if lens else 0}, max {lens[-1] if lens else 0}")


if __name__ == "__main__":
    main()
