## Environment

* use the installed env 

  ```bash
  conda activate 3dgs-avatar
  ```

* or install from scratch: cannot use `environment.yml` directly.

  First comment out the line `- tinycudann==1.7` in `enviroment.yml`

  ```bash
  conda create --prefix /cluster/courses/digital_humans/datasets/team_8/miniconda3/envs/3dgs-avatar python=3.7.13
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 pytorch-cuda=11.6 -c pytorch -c nvidia
  # installing the following modules requires GPU runtime
  srun --account digital_humans --time=01:00:30 --gpus=1 --pty bash 
  # install tiny cuda nn
  pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
  # build submodules "diff-gaussian-rasterization" and "simple-knn":
  pip install submodules/diff-gaussian-rasterization
  pip install submodules/simple-knn
  # install the rest of the dependencies
  conda env update -p /cluster/courses/digital_humans/datasets/team_8/miniconda3/envs/3dgs-avatar --file environment.yml --prune
  ```

## Training / testing

* Register an account in [wandb](https://wandb.ai/gaussian-splatting-avatar-test) and log in via bash.

* Create a project, e.g. <b>gaussian-splatting-avatar-test</b>.

* change the project name in line 313 of `train.py` and comment out `entity='fast-avatar'`

  ```python
  wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='gaussian-splatting-avatar-test',
        # entity='fast-avatar',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )
  ```

* change the dataset in `main()` of `train.py`, e.g. use ZJUMocap in ARAH format.

  ```python
  config.dataset.root_dir = '/cluster/courses/digital_humans/datasets/team_8/ZJUMoCap'
  ```

* training using ZJUMoCap dataset
srun --account digital_humans --time=05:00:30 --gpus=1 --pty bash
conda activate 3dgs-avatar
python train.py dataset=zjumocap_377_mono
python train.py dataset=zjumocap_393_mono_half
python train.py dataset=zjumocap_387_4views

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
python render.py mode=test dataset.test_mode=view dataset=zjumocap_387_4views_eval

```

reconstruction:
first change the dataset config:
```yaml
  reconstruct_views: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
  # means the first frame!
  reconstruct_frames: [300, 301, 1]
```

```python
python render.py mode=reconstruct dataset.test_mode=view dataset=zjumocap_393_mono_eval_novel
```
