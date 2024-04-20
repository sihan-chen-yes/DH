### Novel_pose evaluation

#### train

Still the same as original, just need to modify the dataset config.

config->dataset->corresponding yaml file, modify:

```yaml
  train_frames: [ 0, 300, 1 ]
  test_frames:
    view: [0, 300, 30]
    video: [0, 300, 1]
    all: [0, 300, 1]
```

Then run:

```Python
python train.py dataset=zjumocap_386_mono_half
```

#### test

config->dataset->corresponding yaml file, modify:

```yaml
  train_frames: [ 0, 300, 1 ]
  val_frames: [ 0, 1, 1 ]
  test_frames:
    view: [300, 0, 1]
    video: [300, 0, 1]
    all: [300, 0, 1]
```

Then run:

```Python
python render.py mode=test dataset.test_mode=view dataset=zjumocap_386_mono_eval_novel
```
