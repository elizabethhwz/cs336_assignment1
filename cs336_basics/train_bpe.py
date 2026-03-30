import regex as re
from collections import Counter, defaultdict
from joblib import Parallel, delayed

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# USE_TINY_SAMPLE = True
# FULL_INPUT_PATH = "/Users/wanzhenhuang/Documents/study/cs336/cs336_assignment1/data/TinyStoriesV2-GPT4-train.txt"
# TINY_INPUT_PATH = "/Users/wanzhenhuang/Documents/study/cs336/cs336_assignment1/tests/fixtures/tiny_bpe_sample.txt"


def split_corpus_by_special_tokens(
    input_path: str, special_tokens: list[str], pattern: str
) -> list[str]:
    """Split the training corpus into chunks around special tokens.

    Args:
        input_path: Path to the BPE training corpus.
        special_tokens: Special tokens that should be preserved as standalone chunks.
        pattern: Regex pattern used later for pretokenization.

    Returns:
        A list of text chunks produced by splitting on the special tokens.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not special_tokens:
        return [text]
    
    pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
    parts = re.split(pattern, text)
    return parts


def initialize_vocab_and_merges(
    special_tokens: list[str], vocab_size: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], int]:
    """Initialize the byte vocabulary, merge list, and remaining merge budget.

    Args:
        special_tokens: Special tokens to append after the base byte vocabulary.
        vocab_size: Target vocabulary size, including special tokens.

    Returns:
        The vocabulary mapping, an empty merge history, and the number of
        merges still available before reaching ``vocab_size``.
    """
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens, start=256):
        vocab[i] = token.encode("utf-8")
    return vocab, [], vocab_size - len(vocab)

def p(part: str, pattern: str) -> Counter[tuple[bytes, ...]]:
    """Pretokenize a text chunk into unique token tuples and count their frequencies.

    Args:
        part Text chunk to pretokenize.
        pattern: Regex pattern used to split the chunk into pretokenized pieces.

    Returns:
        A Counter mapping each unique token tuple to its frequency.
    """
    tokenized = []
    for match in re.finditer(pattern, part):
        token_tuple = tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
        tokenized.append(token_tuple)
    return Counter(tokenized)

def pretokenizer(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """Pretokenize the entire corpus in parallel and aggregate unique token tuples.

    Args:
        input_path: Path to the BPE training corpus.
        special_tokens: Special tokens that should be preserved as standalone chunks.

    Returns:
        A dictionary mapping each unique token tuple across the corpus to its
        total frequency.
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    parts = split_corpus_by_special_tokens(input_path, special_tokens, PAT)
    parts = [part for part in parts if part not in special_tokens]
    results = Parallel(n_jobs=-9)(delayed(p)(part, PAT) for part in parts)
    total_freq = Counter()
    for result in results:
        total_freq.update(result)
    return total_freq

def initialize_word_and_pair_stats(
    total_freq: dict[tuple[bytes, ...], int]
) -> tuple[
    dict[int, tuple[bytes, ...]],
    Counter[int],
    Counter[tuple[bytes, bytes]],
    dict[tuple[bytes, bytes], set[int]],
]:
    """Pretokenize a chunk and build deduplicated word and pair statistics.

    Args:
        total_freq: A dictionary mapping each unique token tuple to its frequency.

    Returns:
        A tuple containing:
        - ``words``: mapping from word id to unique token tuple
        - ``words_freq``: frequency of each word id in the chunk
        - ``pair_freq``: weighted frequency of each adjacent token pair
        - ``pair_occurrences``: word ids containing each adjacent token pair
    """
    words: dict[int, tuple[bytes, ...]] = {}
    words_freq: Counter[int] = Counter()
    next_id = 0

    for token_tuple, freq in total_freq.items():
        words[next_id] = token_tuple
        words_freq[next_id] = freq
        next_id += 1

    pair_freq: Counter[tuple[bytes, bytes]] = Counter()
    pair_occurrences: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for word_id, tokens in words.items():
        freq = words_freq[word_id]
        for pair in zip(tokens, tokens[1:]):
            pair_freq[pair] += freq
            pair_occurrences[pair].add(word_id)

    return words, words_freq, pair_freq, pair_occurrences

def get_most_frequent_pair(pair_freq: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    """Identify the most frequent adjacent token pair.

    Args:
        pair_freq: A Counter mapping adjacent token pairs to their weighted frequencies.            
    Returns:
        The most frequent adjacent token pair, breaking ties by lexicographical order.
    """
    return max(pair_freq.items(), key=lambda item: (item[1], item[0]))[0]

def apply_merge(
    highest_pair: tuple[bytes, bytes],
    words: dict[int, tuple[bytes, ...]],
    words_freq: Counter[int],
    pair_occurrences: dict[tuple[bytes, bytes], set[int]],
    pair_freq: Counter[tuple[bytes, bytes]],
) -> tuple[
    dict[int, tuple[bytes, ...]],
    Counter[int],
    dict[tuple[bytes, bytes], set[int]],
    Counter[tuple[bytes, bytes]],
]:
    """Apply one merge step using the currently most frequent adjacent pair.

    Args:
        highest_pair: The adjacent token pair to merge.
        words: Mapping from word id to the current token tuple for that word type.
        words_freq: Frequency of each word id in the corpus chunk.
        pair_occurrences: Word ids containing each adjacent token pair.
        pair_freq: Weighted frequency of each adjacent token pair.

    Returns:
        Updated versions of the input data structures reflecting the merge.
    """
    merged_token = highest_pair[0] + highest_pair[1]
    id_list = list(pair_occurrences[highest_pair])
    word_to_id = {tokens: word_id for word_id, tokens in words.items()}

    for tokens_id in id_list:
        tokens = words[tokens_id]
        old_tokens = tokens
        freq = words_freq[tokens_id]

        # merge the highest pair in the token tuple
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == highest_pair:
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_id = tokens_id
        new_tokens_tuple = tuple(new_tokens)
        if word_to_id.get(new_tokens_tuple) is not None:
            # If the new token already exists in the vocabulary, update its frequency
            existing_id = word_to_id.get(new_tokens_tuple)
            words_freq[existing_id] += freq
            del words[tokens_id]
            del words_freq[tokens_id]
            new_id = existing_id
        else:
            words[new_id] = new_tokens_tuple

        for pair in zip(tokens, tokens[1:]):
            pair_freq[pair] -= freq
            pair_occurrences[pair].discard(tokens_id)
            if pair_freq[pair] <= 0:
                del pair_freq[pair]
            if not pair_occurrences[pair]:
                del pair_occurrences[pair]

        del word_to_id[old_tokens]

        for pair in zip(new_tokens, new_tokens[1:]):
            pair_freq[pair] += freq
            pair_occurrences[pair].add(new_id)

    return words, words_freq, pair_occurrences, pair_freq


