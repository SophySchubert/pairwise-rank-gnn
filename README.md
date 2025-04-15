# Pairwise ranking GNN
This repo contains the code from my master's thesis: "Improving pairwise ranking problems with graph neural networks".

## Installation
Works with python version 3.9.21 and torch with cuda 11.8
**Due to a change in the torch load api since version 2.5 you probably need to add "weights_only=False" to OGB dataset loader!**
```bash
  pip install -r requirements.txt
```

## Usage
- choose a config from src/config OR
- create a config file and place it in the src/config folder
```bash
  python src/experiment.py src/config/<CONFIG_NAME>.yml
```

## experiments - TODO
default => 1, fc_weight => 2, fc_extra => 3
model_units=64 => 1, 32 => 2
learnrate=0.01 => 1, 0.001 => 2
seed=42 => 42, 43 => 43

| Task      | model_units | Learnrate | seed | Status  | Folder            |
|---------|--------------|-----------|------|--------|---------------|
| default   | 64          | 0.01      | 42 | &#9745; | 2025-03-16-10-26-29 |
| default   | 64          | 0.01      | 43 | &#9745; | 2025-03-16-11-44-17 |
| default   | 64          | 0.001     | 42 | &#9745; | 2025-03-16-13-03-45 |
| default   | 64          | 0.001     | 43 | &#9745; | 2025-03-16-14-10-28 |
| default   | 32          | 0.01      | 42 | &#9745; | 2025-03-16-16-01-32 |
| default   | 32          | 0.01      | 43 | &#9744; | - |
| default   | 32          | 0.001     | 42 | gerade | 2025-03-16-17-19-30 |
| default   | 32          | 0.001     | 43 | &#9744; | - |
| fc_weight | tbd         | tbd     | tbd   | &#9744; | - |
| fc_extra  | tbd         | tbd     | tbd   | &#9744; | - |

| Unchecked | Checked |
| --------- | ------- |
| &#9744;   | &#9745; |

| #Conv                | #FC   | Akt. Fkt.             | #Neurons   | Batchsize   | LR     | Dropout   | Epochen   | Ordner                | Ergebnis      |
|----------------------|-------|-----------------------|------------|-------------|--------|-----------|-----------|-----------------------|---------------|
| 4  gcnconv     default      | 2     | TanH  global mean     | 256        | 32          | 1e-2   | 0         | 201       | 2025-03-29-11-55-10   | nein          |
| 4   gcnconv    default      | 3     | TanH  global mean     | 64         | 32          | 1e-3   | 0         | 201       | 2025-03-29-12-30-39   | nein          |
| 4   gcnconv    default      | 3     | TanH  global mean     | 64         | 32          | 1e-3   | 0.3       | 201       | 2025-03-29-12-54-18   | nein          |
| 3   gcnconv     default     | 3     | TanH  global mean     | 32         | 32          | 1e-3   | 0.5       | 201       | 2025-03-29-13-54-55   | weniger       |
| 3   gcnconv     default     | 3     | TanH  global mean     | 32         | 32          | 1e-3   | 0.5       | 201       | 2025-03-29-14-51-57   | kinda         |
| 3    gcnconv    default     | 3     | TanH  global mean     | 32         | 32          | 1e-3   | 0.5       | 201       | 2025-03-29-15-39-44   | yes           | 
| 3    gcnconv    default     | 3     | TanH  global mean     | 40         | 32          | 1e-3   | 0.7       | 201       | 2025-03-29-17-47-08   | eher weniger  | 
| 3   gcnconv     default     | 3     | TanH  global mean     | 50         | 32          | 1e-3   | 0.5       | 201       | 2025-03-29-19-34-38   | eher weniger  |
| 3   gcnconv     default     | 3     | TanH  global mean     | 32         | 32          | 1e-3   | 0.4       | 201       | 2025-03-30-19-37-57   | nein          |
| 3   gcnconv     default     | 3     | TanH  global max      | 32         | 32          | 1e-3   | 0.5       | 201       | 2025-03-30-20-56-29   |               |
| 2   gcnconv     default     | 3     | TanH  global max      | 32         | 32          | 1e-3   | 0.5       | 201       | 2025-03-30-22-36-03   | weniger       |
| 7    gcnconv    default     | 2     | TanH  global mean     | 64         | 32          | 1e-3   | 0.5       | 201       | 2025-03-31-19-21-55   | kinda         |
| 5    gcnconv    default     | 2     | TanH  global mean     | 32         | 32          | 1e-3   | 0.25      | 201       | 2025-04-1-09-03-13    | nein          |
| 5   gcnconv      default    | 2     | TanH  global mean     | 64         | 32          | 1e-3   | 0.5       | 201       | 2025-04-1-10-42-05    | weniger       |
| 5    gcnconv    default     | 2     | TanH  global mean     | 32         | 32          | 1e-3   | 0.75      | 201       | 2025-04-1-11-35-30    | weniger       |
| 3   gcnconv     default     | 3     | TanH  global mean     | 32         | 32          | 1e-3   | 0.8       | 201       | 2025-04-01-16-38-00   | nein          | 
| 3   gcnconv     default     | 3     | TanH  global mean     | 32         | 32          | 1e-3   | 0.5       | 201       | 2025-04-01-18-27-33   | weniger       | 
| -------              | ----- | ---------------------- | ---------- | ----------- | ------ | --------- | --------- | --------------------- | ------------- |
| 2    gcnconv    default     | 3     | TanH  global mean     | 32         | 32          | 0.001  | 0.5       | 501       | 2025-04-02-11-38-16   | evtl          | 
| 2   gcnconv     default     | 3     | TanH  global mean     | 32         | 128         | 0.001  | 0.5       | 201       | 2025-04-05-16-14-39   | nein          | 
| 2   gcnconv     default     | 3     | TanH  global mean     | 32         | 512         | 0.005  | 0.5       | 201       | 2025-04-06-19-49-01   | evtl          |
| 2   gcnconv     default     | 3     | TanH  global mean     | 32         | 512         | 0.005  | 0.6       | 201       | 2025-04-06-19-56-11   | nein          | 
| 2   gcnconv     default     | 3     | TanH  global mean     | 32         | 512         | 0.005  | 0.4       | 201       | 2025-04-06-20-10-55   | nein          |
| 2    gcnconv    default     | 3     | TanH  global mean     | 32         | 512         | 0.005  | 0.3       | 201       | 2025-04-06-20-19-48   | nein          |
| 2   gcnconv     default     | 3     | TanH  global mean     | 32         | 512         | 0.008  | 0.3       | 201       | 2025-04-06-20-48-38   | nein          |
| 2   gcnconv      default    | 3     | TanH  global mean     | 32         | 512         | 0.01   | 0.3       | 201       | 2025-04-06-20-48-29   | nein          |
| 3   gcnconv    default      | 3     | TanH  global mean     | 40         | 512         | 0.003  | 0.3       | 301       | 2025-04-06-21-17-46   | nein          |
| 3   gcnconv      default    | 3     | TanH  global mean     | 32         | 512         | 0.003  | 0.5       | 301       | 2025-04-06-21-35-53   | weniger       |
| 3   gcnconv      default    | 3     | TanH  global mean     | 32         | 512         | 0.005  | 0.5       | 301       | 2025-04-06-21-46-33   | nein          |
| 3    gcnconv     default    | 3     | TanH  global mean     | 32         | 512         | 0.008  | 0.5       | 301       | 2025-04-06-22-05-02   | nein          |
| 3   gcnconv      default    | 3     | TanH  global mean     | 32         | 256         | 0.003  | 0.5       | 301       | 2025-04-06-22-17-57   | evlt          |
| 3   gcnconv      default    | 3     | TanH  global mean     | 32         | 256         | 0.003  | 0.5       | 301       | 2025-04-06-22-47-20   | evlt          |
| 2   gcnconv      default    | 3     | TanH  global mean     | 32         | 512         | 0.001  | 0.5       | 501       | 2025-04-08-11-36-54   | yes           |
| 3   gcnconv     default     | 3     | TanH  global mean     | 32         | 512         | 0.001  | 0.5       | 501       | 2025-04-08-11-57-46   | kinda         |
| 4   gcnconv     default     | 3     | TanH  global mean     | 32         | 512         | 0.001  | 0.5       | 501       | 2025-04-08-19-45-22   | kinda         |
| 4   gcnconv  default | 3     | TanH  global mean     | 32         | 1024        | 0.001  | 0.5       | 501       | 2025-04-08-20-09-26   | nein          |
| 2  gcnconv      default     | 3     | TanH  global mean     | 32         | 512         | 0.003  | 0.5       | 501       | 2025-04-08-20-21-37   | nein          |
| 3  gatconv      default     | 3     | TanH  global mean     | 32         | 512         | 0.001  | 0.5       | 501       | 2025-04-08-20-41-34   |  nein             |