#!/usr/bin/env python3
"""Open-MOPD E2E — Step 10: synthesize every artifact the pipeline needs.

Nothing here touches the network or any HuggingFace cache. Everything is built
from scratch so the E2E run is reproducible on a fresh worker.

Produces, under --out (default /data/ycfeng/tmp/openmopd-e2e/assets):

    tokenizer/                 ByteLevel-BPE tokenizer, vocab 1024, Qwen-style chat template
    models/student/            tiny Qwen3ForCausalLM, seed 0
    models/teacher_math/       tiny Qwen3ForCausalLM, seed 1
    models/teacher_code/       tiny Qwen3ForCausalLM, seed 2
    models/teacher_if/         tiny Qwen3ForCausalLM, seed 3
    data/sft_train.parquet     columns: messages, domain, source, dataset, split,
                               difficulty, sample_id
    data/sft_val.parquet
    data/rl_train.parquet      columns: prompt, data_source, ability,
                               reward_model, extra_info, domain
    data/rl_val.parquet
    data/eval_input.parquet    columns: prompt, dataset, reward_model, repeat_idx

SFT schema evidence (this is NOT the sft_trainer.yaml default, and the difference
matters): scripts/local/sft.sh launches `-m verl.trainer.sft_trainer`, whose Hydra
entry is `@hydra.main(config_name="sft_trainer_engine")` (verl/trainer/sft_trainer.py:378)
— so sft_trainer.yaml with its prompt_key=question / response_key=answer is a
DIFFERENT trainer (fsdp_sft_trainer.py) and is not in play. sft_trainer.py always
builds a MultiTurnSFTDataset (verl/trainer/sft_trainer.py:393) unless
data.custom_cls.path is set, and MultiTurnSFTDataset reads its column name from
config["multiturn"]["messages_key"] (verl/utils/dataset/multiturn_sft_dataset.py:63-66).
sft_trainer_engine.yaml declares messages_key FLAT under `data:` (line 26), which that
nested lookup never sees, so the effective column name is the hard default "messages".

Design notes that matter:

* head_dim is pinned to 64, not hidden_size/num_heads. Attention backends in vLLM
  and FlashAttention only accept head dims from a fixed set; 16 or 32 gets rejected
  at engine start. hidden_size stays 128 and q_proj maps 128 -> num_heads*64.
* Transformers is explicitly pinned to the eager attention backend. This keeps the
  dummy run independent of flash-attn, which is intentionally omitted from setup.
* The four models share ONE tokenizer but use different torch seeds, so the
  teachers genuinely disagree. That is what makes
  mt_opd/conflict/* and mt_opd/domain/*/reward_abs_mean non-degenerate.
* model vocab_size == tokenizer vocab_size exactly. A larger model vocab lets the
  sampler emit ids the tokenizer cannot decode.
* The `domain` column is the MT-OPD routing key
  (verl/trainer/ppo/ray_trainer.py reads non_tensor_batch["domain"]). It is a plain
  extra parquet column: RLHFDataset returns the whole row (rl_dataset.py:338-497)
  and collate_fn funnels every non-tensor column into non_tensor_batch
  (rl_dataset.py:96-111).
* data_source values are chosen so validation stays runnable without real benchmark
  payloads: verl/utils/reward_score/opd_val_dispatch.py routes anything that is not
  IF-like or LCB-like to ttrl_math. Exercising the real IF/LCB verifier branches
  needs the actual benchmark metadata and is out of scope for a dummy run.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]

# A small synthetic corpus. Only job: give the BPE trainer enough material to
# produce merges on top of the 256-symbol byte alphabet.
CORPUS_SEEDS = [
    "Solve the problem step by step and put the final answer in \\boxed{}.",
    "What is 2 + 2? The answer is \\boxed{4}.",
    "def solve(n):\n    return n * 2\n",
    "Write a python function that returns the sum of a list.",
    "Answer in exactly three sentences. Do not use the word banana.",
    "The assistant follows every instruction precisely and formats output as JSON.",
    "user assistant system content role message prompt response token",
    "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
    "math code instruction following distillation teacher student policy reward",
    "Let x = 7. Then 3x + 1 = 22, so the answer is \\boxed{22}.",
]


def build_corpus(n: int = 4000) -> list[str]:
    rng = random.Random(1234)
    out = list(CORPUS_SEEDS)
    words = " ".join(CORPUS_SEEDS).split()
    for _ in range(n):
        k = rng.randint(4, 24)
        out.append(" ".join(rng.choice(words) for _ in range(k)))
    return out


def make_tokenizer(out_dir: Path, vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(build_corpus(), trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
        unk_token="<|endoftext|>",
        chat_template=CHAT_TEMPLATE,
        model_max_length=4096,
        clean_up_tokenization_spaces=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(out_dir))
    print(f"[10] tokenizer vocab_size={fast.vocab_size} -> {out_dir}")
    return fast


def make_model(out_dir: Path, tokenizer, seed: int, hidden: int, layers: int):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    vocab = len(tokenizer)
    cfg = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,  # pinned: attention backends reject small head dims
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=tokenizer.convert_tokens_to_ids("<|pad|>"),
        dtype="bfloat16",
        attn_implementation="eager",
        sliding_window=None,
        use_sliding_window=False,
    )
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    # Transformers omits the internal attention selector when serializing a
    # PretrainedConfig. Persist the public field so from_pretrained() keeps the
    # eager backend on CUDA workers as well.
    config_path = out_dir / "config.json"
    config_data = json.loads(config_path.read_text())
    config_data["attn_implementation"] = "eager"
    config_path.write_text(json.dumps(config_data, indent=2) + "\n")
    tokenizer.save_pretrained(str(out_dir))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[10] model seed={seed} params={n_params/1e6:.3f}M vocab={vocab} -> {out_dir}")
    return n_params


# ---------------------------------------------------------------- data

MATH_Q = "Compute {a} + {b}. Put the final answer in \\boxed{{}}."
CODE_Q = "Write a python function add_{a}(x) that returns x + {a}. Return only code."
IF_Q = "Reply with exactly {a} words. Do not use the letter z."

DOMAIN_SPEC = [
    # domain label, data_source, ability, question template
    ("math", "math_dapo", "math", MATH_Q),
    ("code", "dummy_code", "code", CODE_Q),
    ("if", "dummy_if", "instruction_following", IF_Q),
]


def _rows(n_per_domain: int, split: str):
    rng = random.Random(7)
    rows = []
    idx = 0
    for domain, data_source, ability, template in DOMAIN_SPEC:
        for _ in range(n_per_domain):
            a, b = rng.randint(1, 40), rng.randint(1, 40)
            question = template.format(a=a, b=b)
            gt = str(a + b) if domain == "math" else str(a)
            rows.append(
                {
                    "prompt": [{"role": "user", "content": question}],
                    "data_source": data_source,
                    "ability": ability,
                    "reward_model": {"style": "rule", "ground_truth": gt},
                    "extra_info": {"index": idx, "split": split, "question": question},
                    "domain": domain,
                }
            )
            idx += 1
    rng.shuffle(rows)
    return rows


def write_data(out_dir: Path, n_per_domain: int):
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- RL / OPD / MT-OPD prompt parquet
    train_rows = _rows(n_per_domain, "train")
    val_rows = _rows(max(1, n_per_domain // 3), "val")
    pd.DataFrame(train_rows).to_parquet(out_dir / "rl_train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(out_dir / "rl_val.parquet", index=False)
    print(f"[10] rl_train.parquet rows={len(train_rows)} "
          f"domains={sorted({r['domain'] for r in train_rows})}")
    print(f"[10] rl_val.parquet   rows={len(val_rows)}")

    # ---- SFT parquet: MultiTurnSFTDataset reads the `messages` column
    #      (see the module docstring for why it is `messages` and not question/answer).
    #      The extra columns mirror the released MixSFT schema so the shape is realistic.
    sft_rows = [
        {
            "messages": [
                {"role": "user", "content": r["extra_info"]["question"]},
                {
                    "role": "assistant",
                    "content": f"Reasoning omitted. The answer is "
                               f"\\boxed{{{r['reward_model']['ground_truth']}}}.",
                },
            ],
            "domain": r["domain"],
            "source": "dummy",
            "dataset": f"dummy_{r['domain']}",
            "split": "train",
            "difficulty": "easy",
            "sample_id": f"{r['domain']}-{r['extra_info']['index']}",
        }
        for r in train_rows
    ]
    pd.DataFrame(sft_rows).to_parquet(out_dir / "sft_train.parquet", index=False)
    pd.DataFrame(sft_rows[: max(1, len(sft_rows) // 4)]).to_parquet(
        out_dir / "sft_val.parquet", index=False
    )
    print(f"[10] sft_train.parquet rows={len(sft_rows)} "
          f"cols=messages,domain,source,dataset,split,difficulty,sample_id")

    # ---- eval input parquet: evals/rollout_engine/vllm_rollout.py needs `prompt`,
    #      groups output files by `dataset`, and the AIME scorer reads reward_model.
    eval_rows = []
    for i, r in enumerate(val_rows):
        eval_rows.append(
            {
                "prompt": r["prompt"],
                "dataset": "dummy_math",
                "reward_model": r["reward_model"],
                "data_source": "math_dapo",
                "repeat_idx": 0,
                "index": i,
            }
        )
    pd.DataFrame(eval_rows).to_parquet(out_dir / "eval_input.parquet", index=False)
    print(f"[10] eval_input.parquet rows={len(eval_rows)} dataset=dummy_math")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/data/ycfeng/tmp/openmopd-e2e/assets"))
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--rows-per-domain", type=int, default=8)
    ap.add_argument("--clean", action="store_true", help="delete --out first")
    args = ap.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = make_tokenizer(args.out / "tokenizer", args.vocab_size)

    models_dir = args.out / "models"
    manifest = {"tokenizer_vocab": len(tokenizer), "models": {}}
    for name, seed in [("student", 0), ("teacher_math", 1), ("teacher_code", 2), ("teacher_if", 3)]:
        n = make_model(models_dir / name, tokenizer, seed, args.hidden, args.layers)
        manifest["models"][name] = {"seed": seed, "params": n, "path": str(models_dir / name)}

    write_data(args.out / "data", args.rows_per_domain)

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[10] manifest -> {args.out / 'manifest.json'}")
    print("[10] done")


if __name__ == "__main__":
    main()
