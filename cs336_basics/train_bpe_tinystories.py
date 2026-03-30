from time import perf_counter
from tests.adapters import run_train_bpe
import pickle
from pathlib import Path

out_dir = Path("/Users/wanzhenhuang/Documents/study/cs336/cs336_assignment1/trained_data")
out_dir.mkdir(exist_ok=True)

start_time = perf_counter()

vocab, merges = run_train_bpe(
    input_path='/Users/wanzhenhuang/Documents/study/cs336/cs336_assignment1/data/TinyStoriesV2-GPT4-train.txt',
    vocab_size=1000,
    special_tokens=["<|endoftext|>"],
)

elapsed_time = perf_counter() - start_time

with open(out_dir / "tinystories_bpe.pkl", "wb") as f:
    pickle.dump(
        {
            "vocab": vocab,
            "merges": merges,
            "special_tokens": ["<|endoftext|>"],
        },
        f,
    )

print("vocab size:", len(vocab))
print("num merges:", len(merges))
print("first 10 merges:", merges[:10])
print(f"execution time: {elapsed_time:.2f} seconds")
