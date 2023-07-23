# DeepDanbooru

This is a pure pytorch implementation of DeepDanbooru neural network: https://github.com/KichangKim/DeepDanbooru/.

The project was inspired by https://github.com/AUTOMATIC1111/TorchDeepDanbooru

The RestNet network was used to optimize the AUTOMATIC1111 code and introduce LSTM with Transformer, and I haven't adequately compared the performance of that improvement with the original KichangKim version.

The original ONNX does not run directly and requires Train training using my new structure.
