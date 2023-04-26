### 实验记录

0.1 比例的预训练数据和全量预训练数据

| Method  | Dataset ratio  |Pretraining epoch  |Finetuning epoch|Finetuning |Voting |
|---|---|---|---|---|---|
| PointMAE(lr 300)  | 1.0 |300|300|93.07%|93.35%|
| PointMAE(lr 300)  | 0.1 |100|300|92.54%|---|
| PointMAE(lr 100)  | 0.1 |100|300|---|---|
| 2-fold EMAE(lr 300)  | 0.1 |100|300|92.79%|93.44%|
| 2-fold EMAE(lr 100)  | 0.1 |100|300|||


| Method  | Dataset ratio  |Pretraining epoch  |Finetuning epoch|Finetuning |Voting |
|---|---|---|---|---|---|
| PointMAE | 1.0 |300 (15h) |300|93.07%|93.35%|
| 2-fold EMAE  | 0.1 |100 (0.5h) |300|92.79%|93.44%|