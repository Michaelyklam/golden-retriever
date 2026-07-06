import json
from pathlib import Path

from golden_retriever.train_lora_sft import load_chat_jsonl, mask_prompt_labels, pad_batch, tokenize_example


class FakeEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.calls.append({"messages": messages, "add_generation_prompt": add_generation_prompt, "kwargs": kwargs})
        if add_generation_prompt and kwargs.get("enable_thinking") is True:
            return "1 2 3"
        if add_generation_prompt:
            return "1 2"
        return "1 2 3 4 5"

    def __call__(self, text, add_special_tokens=False):
        return FakeEncoding([int(part) for part in text.split()])


def test_load_chat_jsonl_reads_messages(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"messages":[{"role":"assistant","content":"ok"}]}\n', encoding="utf-8")

    examples = load_chat_jsonl(path)

    assert examples == [{"messages": [{"role": "assistant", "content": "ok"}]}]


def test_mask_prompt_labels_masks_only_prompt_tokens():
    labels = mask_prompt_labels(input_ids=[10, 11, 12, 13], prompt_length=2)

    assert labels == [-100, -100, 12, 13]


def test_tokenize_example_can_align_prompt_with_thinking_enabled():
    tokenizer = FakeTokenizer()
    example = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}

    tokenized = tokenize_example(example, tokenizer, max_length=16, enable_thinking=True)

    assert tokenized["input_ids"] == [1, 2, 3, 4, 5]
    assert tokenized["labels"] == [-100, -100, -100, 4, 5]
    assert tokenizer.calls[1]["kwargs"] == {"enable_thinking": True}


def test_pad_batch_pads_ids_attention_and_labels():
    batch = pad_batch(
        [
            {"input_ids": [1, 2], "labels": [-100, 2]},
            {"input_ids": [3], "labels": [3]},
        ],
        pad_token_id=0,
    )

    assert batch["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]
