# Raw corpora

## `tinyshakespeare.txt`

The concatenated works of Shakespeare (~1.1 MB, ~1.1M characters), the standard
small-scale language-modelling corpus popularised by Karpathy's `char-rnn`.

Shakespeare's text is in the **public domain**, which is why it can be committed
directly and why this repository needs no download step to be useful.

Source: <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>

To use a different corpus, point `lmops prepare` at any UTF-8 text file:

```bash
lmops prepare --input path/to/your.txt --out data/processed --vocab-size 4096
```
