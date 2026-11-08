# Teach gpt-oss-20b chess

Say you want [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) to be good at chess. This example follows what you would do: measure how good it is, build a dataset, fine-tune it, and measure again.

## 1. Is gpt-oss-20b good at chess?

Measure before training. [Lichess puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles) are positions from real games where one move wins, by mating or winning material, and each one is rated by how hard Lichess players found it. We hold out 1,000 of them as a test set, with a median rating of 1390.

[`evaluate.py`](evaluate.py) shows the model each test position and checks its move. It reports the share of puzzles solved and a puzzle rating: the Lichess rating at which the [Elo formula](https://en.wikipedia.org/wiki/Elo_rating_system) expects a player to solve as many of these puzzles.

```sh
python examples/sft_gpt_oss_chess/evaluate.py openai/gpt-oss-20b --reasoning_effort low
```

```
openai/gpt-oss-20b: 14.8% of the puzzles solved, puzzle rating 764
```

gpt-oss-20b reasons before it answers. At its default reasoning effort, most of its answers do not finish within 8k tokens, hence the low effort. It misses more than 3 in 4 of the puzzles rated under 1000. Its long reasoning also makes the score move from run to run: three runs gave 11.7%, 14.8% and 15.3%.

## 2. Build the dataset

The Lichess database holds 6 million puzzles. Here is one of them:

```python
>>> from datasets import load_dataset
>>> dataset = load_dataset("Lichess/chess-puzzles", split="train")
>>> dataset.filter(lambda row: row["PuzzleId"] == "stVYb")[0]
{'PuzzleId': 'stVYb', 'GameId': 'oiifHiFU/black#20', 'FEN': 'r2q1rk1/ppp2pbp/3pbnp1/2nN2N1/2B1P3/5Q2/PPP2PPP/R1B1K2R b KQ - 1 10', 'Moves': 'e6g4 d5f6 d8f6 f3g4', 'Rating': 1419, 'RatingDeviation': 75, 'Popularity': 97, 'NbPlays': 18322, 'Themes': ['crushing', 'fork', 'opening', 'short'], 'OpeningTags': ['Pirc_Defense', 'Pirc_Defense_Kholmov_System']}
```

`FEN` is the position one move before the puzzle, and `Moves` lists, in [UCI notation](https://en.wikipedia.org/wiki/Universal_Chess_Interface) (from-square then to-square), the opponent's move that sets up the puzzle, then the solution. To train on it, the puzzle becomes a prompt-completion conversation: the prompt shows the board after the opponent's move, its FEN and the legal moves, and the completion is the first move of the solution. [`chess_puzzles.py`](../datasets/chess_puzzles.py) does it with a `map`:

```python
import chess


def to_prompt_completion(example):
    board = chess.Board(example["FEN"])
    moves = example["Moves"].split()
    board.push_uci(moves[0])  # the opponent's move that sets up the puzzle
    side = "White" if board.turn == chess.WHITE else "Black"
    legal_moves = " ".join(sorted(move.uci() for move in board.legal_moves))
    prompt = (
        f"Chess puzzle. {side} to move.\n\n{board}\n\nFEN: {board.fen()}\nLegal moves (UCI): {legal_moves}\n\n"
        "Find the best move. Answer with the move in UCI notation only."
    )
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": moves[1]}],
    }


dataset = dataset.map(
    to_prompt_completion,
    remove_columns=["GameId", "FEN", "Moves", "RatingDeviation", "Popularity", "NbPlays", "OpeningTags"],
)
dataset = dataset.train_test_split(test_size=1_000, seed=0)
```

The same puzzle after the `map`:

```python
{'PuzzleId': 'stVYb', 'Rating': 1419, 'Themes': ['crushing', 'fork', 'opening', 'short'], 'prompt': [{'role': 'user', 'content': 'Chess puzzle. White to move.\n\nr . . q . r k .\np p p . . p b p\n...'}], 'completion': [{'role': 'assistant', 'content': 'd5f6'}]}
```

where the content of the prompt reads:

```
Chess puzzle. White to move.

r . . q . r k .
p p p . . p b p
. . . p . n p .
. . n N . . N .
. . B . P . b .
. . . . . Q . .
P P P . . P P P
R . B . K . . R

FEN: r2q1rk1/ppp2pbp/3p1np1/2nN2N1/2B1P1b1/5Q2/PPP2PPP/R1B1K2R w KQ - 2 11
Legal moves (UCI): a1b1 a2a3 a2a4 b2b3 b2b4 c1d2 c1e3 c1f4 c2c3 c4a6 c4b3 c4b5 c4d3 c4e2 c4f1 d5b4 d5b6 d5c3 d5c7 d5e3 d5e7 d5f4 d5f6 e1d1 e1d2 e1e2 e1f1 e1g1 e4e5 f3a3 f3b3 f3c3 f3d1 f3d3 f3e2 f3e3 f3f4 f3f5 f3f6 f3g3 f3g4 f3h3 g2g3 g5e6 g5f7 g5h3 g5h7 h1f1 h1g1 h2h3 h2h4

Find the best move. Answer with the move in UCI notation only.
```

The converted dataset, with its 1,000 test puzzles split off, is on the Hub as [trl-lib/chess-puzzles](https://huggingface.co/datasets/trl-lib/chess-puzzles).

## 3. Fine-tune

```sh
accelerate launch --config_file examples/accelerate_configs/fsdp2.yaml --num_processes 4 \
    examples/sft_gpt_oss_chess/sft_gpt_oss_chess.py
```

[`sft_gpt_oss_chess.py`](sft_gpt_oss_chess.py) fine-tunes all the weights on 500,000 puzzles for one epoch, about 2 hours on 4xH200. FSDP2 shards the weights, their gradients and the optimizer states across the GPUs.

![Token accuracy during training](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_gpt_oss_chess/token_accuracy.png)

The logged loss stays above 3.6 even at this accuracy: gpt-oss weights the load-balancing loss of its router by 0.9 (`router_aux_loss_coef` in its config), and that loss sits near 4 when the experts are balanced.

## 4. What is particular about gpt-oss: MXFP4

gpt-oss-20b stores the weights of its experts, about 90% of its parameters, in MXFP4: 4-bit values with one shared scale per block of 32. That is why the checkpoint takes 14 GB. Training needs more precision than that, so the model makes a round trip:

- **Dequantized to train.** `SFTTrainer(model="openai/gpt-oss-20b")` sees the MXFP4 checkpoint and loads its experts dequantized, in float32 like the rest of the model.
- **Quantized again to serve.** At the end, the script loads the fine-tuned model with `Mxfp4Config()`, which quantizes the experts as they load, and saves it in the format of the original checkpoint:

```python
model = AutoModelForCausalLM.from_pretrained(
    training_args.output_dir, quantization_config=Mxfp4Config(), dtype=torch.bfloat16, device_map="cuda"
)
model.save_pretrained(f"{training_args.output_dir}-mxfp4")
```

So the run writes two models: `gpt-oss-20b-chess` in float32 (84 GB), and `gpt-oss-20b-chess-mxfp4` (14 GB), which serves like the original.

## 5. Measure again

```sh
python examples/sft_gpt_oss_chess/evaluate.py gpt-oss-20b-chess-mxfp4
```

```
gpt-oss-20b-chess-mxfp4: 69.9% of the puzzles solved, puzzle rating 1813
```

| Model | Puzzles solved | Puzzle rating |
| --- | --- | --- |
| gpt-oss-20b, low reasoning effort | 14.8% | 764 |
| Fine-tuned (float32) | 71.8% | 1855 |
| Fine-tuned, back in MXFP4 | 69.9% | 1813 |

The fine-tuned model answers right away, without reasoning, and solves nearly 5 times more puzzles. Its score does not move between runs: it answers in a few tokens. Back in MXFP4 it loses 1.9 points. Training on more puzzles helps, and the MXFP4 model keeps most of the gain at every size:

![Puzzles solved against the number of training puzzles](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_gpt_oss_chess/puzzles_solved.png)

Each number is one training run on the same 1,000 test puzzles, and in our runs the same settings trained twice differed by 3 points. Training with the MXFP4 rounding in the forward pass, quantization-aware training, would remove the loss from re-quantizing: see [#6567](https://github.com/huggingface/trl/issues/6567).

Two things to keep in mind about the puzzle rating. It counts a puzzle as solved on its first move, while Lichess asks for the whole solution, so it is higher than the rating the model would get on Lichess. And it measures tactics, not how well the model plays a full game.

Here are three test puzzles that the original model missed in two runs and the fine-tuned model solves in MXFP4. The red arrow is the original model's move in one run, the green arrow the fine-tuned model's move:

<table>
  <tr>
    <td><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_gpt_oss_chess/puzzle-stVYb.png" width="240" alt="White to move: Nxf6+"></td>
    <td><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_gpt_oss_chess/puzzle-fBl2f.png" width="240" alt="Black to move: Nxf3+"></td>
    <td><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_gpt_oss_chess/puzzle-rsjTA.png" width="240" alt="Black to move: Rxg2+"></td>
  </tr>
  <tr>
    <td><a href="https://lichess.org/training/stVYb">White to move</a>, rated 1419. 11. Nxf6+ Qxf6 12. Qxg4: the capture with check removes the knight that guards the bishop on g4, and the queen takes it. The original model played Qg3.</td>
    <td><a href="https://lichess.org/training/fBl2f">Black to move</a>, rated 1183. 15...Nxf3+ 16. gxf3 Qxh2#: the knight leaves e5 with check and opens the queen's diagonal to h2. The original model played Qd2.</td>
    <td><a href="https://lichess.org/training/rsjTA">Black to move</a>, rated 1818. 23...Rxg2+ 24. Kxg2 Bc6+ 25. Kf2 Qxf7: the rook sacrifice brings the king to g2, and the bishop's check uncovers the queen, which takes White's queen. The original model played Qc6.</td>
  </tr>
</table>

## 6. Serve it

The MXFP4 model loads like the original, in vLLM or transformers:

```sh
vllm serve gpt-oss-20b-chess-mxfp4
```

The model from the run above is on the Hub as [trl-lib/gpt-oss-20b-chess](https://huggingface.co/trl-lib/gpt-oss-20b-chess), to try it without training:

```sh
vllm serve trl-lib/gpt-oss-20b-chess
```
