import torch

from latent_context.processor import LCLMProcessor


class TinyDecoderTokenizer:
    _special = {
        "<|memory_start|>": 101,
        "<|memory_end|>": 102,
        "<|memory|>": 103,
    }

    def __init__(self):
        self.pad_token_id = 0
        self._char_ids = {}
        self._inverse = {value: key for key, value in self._special.items()}

    def convert_tokens_to_ids(self, token):
        return self._special.get(token)

    def convert_ids_to_tokens(self, token):
        if isinstance(token, list):
            return [self.convert_ids_to_tokens(item) for item in token]
        return self._inverse.get(token, f"char-{token}")

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        cursor = 0
        specials = sorted(self._special, key=len, reverse=True)
        while cursor < len(text):
            match = next((token for token in specials if text.startswith(token, cursor)), None)
            if match is not None:
                ids.append(self._special[match])
                cursor += len(match)
                continue
            char = text[cursor]
            if char not in self._char_ids:
                token_id = 1000 + len(self._char_ids)
                self._char_ids[char] = token_id
                self._inverse[token_id] = char
            ids.append(self._char_ids[char])
            cursor += 1
        return ids

    def __call__(
        self,
        texts,
        *,
        padding="longest",
        truncation=False,
        max_length=None,
        add_special_tokens=False,
        return_tensors="pt",
    ):
        del return_tensors
        encoded = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        width = max(len(ids) for ids in encoded) if padding else None
        padded = []
        masks = []
        for ids in encoded:
            pad = (width - len(ids)) if width is not None else 0
            padded.append(ids + [self.pad_token_id] * pad)
            masks.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class TinyEmbedTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(1, len(text) + 1))


def make_processor(compression_ratio=2):
    return LCLMProcessor(
        decoder_tokenizer=TinyDecoderTokenizer(),
        embed_tokenizer=TinyEmbedTokenizer(),
        compression_ratio=compression_ratio,
    )


def test_target_cot_memory_is_encoded_but_not_supervised():
    processor = make_processor(compression_ratio=2)
    prompt = "USER: question\nASSISTANT: "
    target = (
        "<|memory_start|>think<|memory_end|>"
        "FINAL"
    )

    result = processor.process_wrapped_batch(
        prompts=[prompt],
        targets=[target],
        padding=False,
        truncation=False,
    )

    # Five embed IDs at ratio two become three decoder latents.
    assert result["latent_counts"] == [[3]]
    assert result["memory_token_ids"] == [[[1, 2, 3, 4, 5]]]
    assert len(result["memory_positions"][0]) == 1
    start, end = result["memory_positions"][0][0]
    assert end - start == 3

    labels = result["labels"][0]
    assert labels[start - 1 : end + 1].tolist() == [-100] * 5
    final_ids = processor.decoder_tokenizer.encode("FINAL", add_special_tokens=False)
    assert labels[-len(final_ids) :].tolist() == final_ids


def test_prompt_and_target_regions_preserve_serialized_metadata_order():
    processor = make_processor(compression_ratio=3)
    prompt = "P<|memory_start|>abc<|memory_end|>Q"
    target = "T<|memory_start|>1234567<|memory_end|>answer"

    result = processor.process_wrapped_batch(
        prompts=[prompt], targets=[target], padding=False, truncation=False
    )

    assert result["memory_token_ids"] == [
        [[1, 2, 3], [1, 2, 3, 4, 5, 6, 7]]
    ]
    assert result["latent_counts"] == [[1, 3]]
    assert [end - start for start, end in result["memory_positions"][0]] == [1, 3]
    assert result["memory_positions"][0][0][0] < result["memory_positions"][0][1][0]

    labels = result["labels"][0]
    for start, end in result["memory_positions"][0]:
        assert labels[start - 1 : end + 1].eq(-100).all()


def test_uncompressed_example_has_no_encoder_metadata():
    processor = make_processor()
    result = processor.process_wrapped_batch(
        prompts=["USER: plain\nASSISTANT: "],
        targets=["plain answer"],
        padding=False,
        truncation=False,
    )

    assert result["memory_token_ids"] == [[]]
    assert result["latent_counts"] == [[]]
    assert result["memory_positions"] == [[]]
    assert result["labels"][0].ne(-100).any()


def test_empty_memory_region_fails_closed():
    processor = make_processor()
    try:
        processor.process_wrapped_batch(
            prompts=["<|memory_start|>   <|memory_end|>"],
            targets=["answer"],
            padding=False,
        )
    except ValueError as error:
        assert "non-whitespace" in str(error)
    else:
        raise AssertionError("empty memory region should be rejected")

