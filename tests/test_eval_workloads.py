"""Tests of the workload builder (CPU only, no download, no transformers): a
whitespace tokenizer and small dataset files stand in for the real ones."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import workloads  # noqa: E402
from eval.common import read_jsonl  # noqa: E402


class WordTokenizer:
    """One token per whitespace-separated word."""

    def __init__(self):
        self.vocab, self.words = {}, []

    def encode(self, text, add_special_tokens=True):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.words)
                self.words.append(w)
            ids.append(self.vocab[w])
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.words[i] for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return f"<user> {messages[0]['content']} <assistant>"


SHAREGPT = [
    {"id": "a", "conversations": [{"from": "human", "value": "hello there"},
                                  {"from": "gpt", "value": "hi"}]},
    {"id": "b", "conversations": [{"from": "gpt", "value": "starts wrong"},
                                  {"from": "human", "value": "x"}]},
    {"id": "c", "conversations": [{"from": "human", "value": "only one"}]},
    {"id": "d", "conversations": [{"from": "human", "value": "   "},
                                  {"from": "gpt", "value": "empty"}]},
    {"id": "e", "conversations": [
        {"from": "human", "value": "first turn " * 5},
        {"from": "gpt", "value": "answer"},
        {"from": "human", "value": "second turn"}]},
]

ITEM = {"_id": "lb1", "context": "  some long document  ",
        "question": " What? ", "choice_A": "one", "choice_B": "two",
        "choice_C": "three", "choice_D": "four"}


def test_sharegpt_takes_the_first_human_turn(tmp_path):
    p = tmp_path / "sg.json"
    p.write_text(json.dumps(SHAREGPT))
    got = workloads.sharegpt_prompts(p)
    assert got == [("sharegpt-0", "hello there"),
                   ("sharegpt-4", "first turn " * 5)]


def test_longbench_template():
    assert workloads.longbench_prompt(ITEM) == (
        "Please read the following text and answer the question below.\n\n"
        "<text>\nsome long document\n</text>\n\n"
        "What is the correct answer to this question: What?\n"
        "Choices:\n(A) one\n(B) two\n(C) three\n(D) four\n\n"
        'Format your response as follows: "The correct answer is '
        '(insert answer here)".')


def candidates(n=50):
    return [(f"r{i}", " ".join(f"w{i}" for _ in range(1 + i * 3)))
            for i in range(n)]


def test_build_is_seeded_and_drops_long_prompts():
    tok = WordTokenizer()
    a = workloads.build(candidates(), tok, 200, max_prompt_tokens=90,
                        num=10, seed=1)
    b = workloads.build(candidates(), tok, 200, max_prompt_tokens=90,
                        num=10, seed=1)
    c = workloads.build(candidates(), tok, 200, max_prompt_tokens=90,
                        num=10, seed=2)
    assert a == b and [r["id"] for r in a] != [r["id"] for r in c]
    assert len(a) == 10
    for r in a:
        assert r["prompt_tokens"] == len(r["prompt"].split()) <= 90
        assert r["max_tokens"] == 200 - r["prompt_tokens"]
    every = workloads.build(candidates(), tok, 200, max_prompt_tokens=90)
    assert len(every) == 30                  # 1 + 3i <= 90 for i < 30
    capped = workloads.build(candidates(), tok, 200, max_prompt_tokens=90,
                             max_output_tokens=16)
    assert all(r["max_tokens"] <= 16 for r in capped)
    # default limit: max_model_len - 1024
    assert workloads.build(candidates(), tok, 1030) == \
        workloads.build(candidates(), tok, 1030, max_prompt_tokens=6)


def test_chat_template_wraps_and_counts():
    tok = WordTokenizer()
    r = workloads.build([("x", "a b c")], tok, 100, max_prompt_tokens=50,
                        chat_template=True)[0]
    assert r["prompt"] == "<user> a b c <assistant>"
    assert r["prompt_tokens"] == 5 and r["max_tokens"] == 95
    with pytest.raises(ValueError):          # default limit: 100 - 1024
        workloads.build([("x", "a b c")], tok, 100)


def test_reference_request_has_1024_tokens():
    tok = WordTokenizer()
    ref = workloads.reference_request([t for _, t in candidates()], tok)
    assert ref["id"] == workloads.REFERENCE_ID
    assert ref["prompt_tokens"] == 1024
    ref = workloads.reference_request([t for _, t in candidates()], tok,
                                      chat_template=True)
    assert ref["prompt_tokens"] == 1024
    with pytest.raises(ValueError):
        workloads.reference_request(["too short"], tok)


def test_main_writes_the_workload_file(tmp_path, monkeypatch):
    items = [dict(ITEM, _id=f"lb{i}", context="word " * (100 * i))
             for i in range(1, 30)]
    p = tmp_path / "data.json"
    p.write_text(json.dumps(items))
    monkeypatch.setattr(workloads, "download", lambda d, c=None: str(p))
    monkeypatch.setattr(workloads, "load_tokenizer",
                        lambda path: WordTokenizer())
    monkeypatch.setattr(sys, "argv", [
        "workloads", "--dataset", "longbench", "--tokenizer", "tok",
        "--max-model-len", "3000", "--max-prompt-tokens", "2000",
        "--results", str(tmp_path / "results"), "--model", "m"])
    workloads.main()
    header, reqs = read_jsonl(tmp_path / "results/m/longbench/workload.jsonl")
    assert header["dataset"] == "longbench" and header["num"] == len(reqs)
    assert header["reference"]["prompt_tokens"] == 1024
    assert 0 < len(reqs) < 29
    assert all(r["prompt_tokens"] <= 2000 and
               r["max_tokens"] == 3000 - r["prompt_tokens"] for r in reqs)
