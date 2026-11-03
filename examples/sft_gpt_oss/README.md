# Fine-tune gpt-oss-20b, then serve it in MXFP4 again

[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) stores the weights of its experts, about 90% of its parameters, in MXFP4: 4-bit values with one shared scale per block of 32. That is why the model fits in 16 GB and serves fast. Training needs more precision than that, so this example:

1. loads gpt-oss-20b with its experts dequantized, which TRL does by default for an MXFP4 checkpoint,
2. fine-tunes all the weights on chess puzzles,
3. quantizes the fine-tuned experts back to MXFP4, so the model serves like the original.

## The task

[Lichess puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles) are positions from real games where one move wins: it mates, or wins material. The model sees the board, its FEN and the legal moves, and answers with the move in [UCI notation](https://en.wikipedia.org/wiki/Universal_Chess_Interface), from-square then to-square. Here the answer is `d5f6`:

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

[trl-lib/chess-puzzles](https://huggingface.co/datasets/trl-lib/chess-puzzles) holds the 6 million Lichess puzzles in this prompt-completion format (see [`chess_puzzles.py`](../datasets/chess_puzzles.py)). The example trains on 500,000 of them and evaluates on 1,000 held-out ones.

## Run it

Fine-tune on one node, about 2 hours on 4xH200:

```sh
accelerate launch --config_file examples/accelerate_configs/fsdp2.yaml --num_processes 4 \
    examples/sft_gpt_oss/sft_gpt_oss.py
```

This writes two models: `gpt-oss-20b-chess`, the fine-tuned model in float32 (84 GB), and `gpt-oss-20b-chess-mxfp4`, the same model with its experts in MXFP4 again (14 GB, the size of the original). Count the puzzles each one solves, on one GPU:

```sh
python examples/sft_gpt_oss/evaluate.py gpt-oss-20b-chess-mxfp4
python examples/sft_gpt_oss/evaluate.py openai/gpt-oss-20b --reasoning_effort low
```

And serve it like the original:

```sh
vllm serve gpt-oss-20b-chess-mxfp4
```

## Results

| Model | Test puzzles solved |
| --- | --- |
| gpt-oss-20b, low reasoning effort | 11.7% |
| Fine-tuned on 500k puzzles | 71.8% |
| Fine-tuned on 500k puzzles, back in MXFP4 | 69.9% |

The fine-tuned model answers right away, without reasoning, and solves 6 times more puzzles than the original model does after reasoning. Back in MXFP4, it loses 1.9 points, and the runs on 50k and 200k puzzles lose 4.3 and 4.4. Each number is one training run on the same 1,000 test puzzles, and in our runs the same settings trained twice differed by 3 points.

![Puzzles solved against the number of training puzzles](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_gpt_oss/puzzles_solved.png)

Training with the MXFP4 rounding in the forward pass, quantization-aware training, is the way to remove that loss: see [#6567](https://github.com/huggingface/trl/issues/6567).

![Token accuracy during training](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_gpt_oss/token_accuracy.png)

The logged loss stays above 3.6 even at this accuracy. gpt-oss weights the load-balancing loss of its router by 0.9 (`router_aux_loss_coef` in its config), and that loss sits near 4 when the experts are balanced, so it accounts for about 3.6 of the logged loss.

## Before and after

Three test puzzles that the original model misses (red arrow) and the fine-tuned model solves in MXFP4 (green arrow):

<table>
  <tr>
    <td><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_gpt_oss/puzzle-stVYb.png" width="240" alt="White to move: Nxf6+"></td>
    <td><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_gpt_oss/puzzle-fBl2f.png" width="240" alt="Black to move: Nxf3+"></td>
    <td><img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_gpt_oss/puzzle-rsjTA.png" width="240" alt="Black to move: Rxg2+"></td>
  </tr>
  <tr>
    <td><a href="https://lichess.org/training/stVYb">White to move</a>, rated 1419. 11. Nxf6+ Qxf6 12. Qxg4: the capture with check removes the knight that guards the bishop on g4, and the queen takes it. The original model plays Qg3.</td>
    <td><a href="https://lichess.org/training/fBl2f">Black to move</a>, rated 1183. 15...Nxf3+ 16. gxf3 Qxh2#: the knight leaves e5 with check and opens the queen's diagonal to h2. The original model plays Qd2.</td>
    <td><a href="https://lichess.org/training/rsjTA">Black to move</a>, rated 1818. 23...Rxg2+ 24. Kxg2 Bc6+ 25. Kf2 Qxf7: the rook sacrifice brings the king to g2, and the bishop's check uncovers the queen, which takes White's queen. The original model plays Qc6.</td>
  </tr>
</table>

## How it works

- **Loading.** `SFTTrainer(model="openai/gpt-oss-20b")` reads the checkpoint's quantization config and loads it with `Mxfp4Config(dequantize=True)`, in float32 like any model TRL loads by name. Pass your own `quantization_config` in `model_init_kwargs` to load it otherwise.
- **Training.** FSDP2 shards the float32 weights, their gradients and the optimizer states across the GPUs, and runs the forward and backward in bfloat16.
- **Back to MXFP4.** `from_pretrained(..., quantization_config=Mxfp4Config())` quantizes the experts as it loads them, and `save_pretrained` writes them as `*_blocks` and `*_scales` with the `quantization_config` of the original checkpoint, so vLLM and transformers load it like the original.
