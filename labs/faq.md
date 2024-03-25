### TOC: FAQ

### TOCEntry: Install

- _Installing to central user packages repository_

  You can install all required packages to central user packages repository using
  `python3 -m pip install --user --no-cache-dir keras~=3.0.5 --extra-index-url=https://download.pytorch.org/whl/cu118 torch~=2.2.0 torchaudio~=2.2.0 torchvision~=0.17.0 torchmetrics~=1.3.1 flashlight-text~=0.0.3 tensorboard~=2.16.2 transformers~=4.37.2 gymnasium~=1.0.0a1 pygame~=2.5.2`.

  The above command installs CUDA 11.8 PyTorch build, but you can change `cu118` to:
  - `cpu` to get CPU-only (smaller) version,
  - `cu121` to get CUDA 12.1 build,
  - `rocm5.7` to get AMD ROCm 5.7 build.

- _Installing to a virtual environment_

  Python supports virtual environments, which are directories containing
  independent sets of installed packages. You can create a virtual environment
  by running `python3 -m venv VENV_DIR` followed by
  `VENV_DIR/bin/pip install --no-cache-dir keras~=3.0.5 --extra-index-url=https://download.pytorch.org/whl/cu118 torch~=2.2.0 torchaudio~=2.2.0 torchvision~=0.17.0 torchmetrics~=1.3.1 flashlight-text~=0.0.3 tensorboard~=2.16.2 transformers~=4.37.2 gymnasium~=1.0.0a1 pygame~=2.5.2`.
  (or `VENV_DIR/Scripts/pip` on Windows).

  Again, apart from the CUDA 11.8 build, you can change `cu118` to:
  - `cpu` to get CPU-only (smaller) version,
  - `cu121` to get CUDA 12.1 build,
  - `rocm5.7` to get AMD ROCm 5.7 build.

- _**Windows** installation_

  - On Windows, it can happen that `python3` is not in PATH, while `py` command
    is – in that case you can use `py -m venv VENV_DIR`, which uses the newest
    Python available, or for example `py -3.11 -m venv VENV_DIR`, which uses
    Python version 3.11.

  - If you encounter a problem creating the logs in the `args.logdir` directory,
    a possible cause is that the path is longer than 260 characters, which is
    the default maximum length of a complete path on Windows. However, you can
    increase this limit on Windows 10, version 1607 or later, by following
    the [instructions](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation).

- _**GPU** support on Linux and Windows_

  PyTorch supports NVIDIA GPU or AMD GPU out of the box, you just need to select
  appropriate `--extra-index-url` when installing the packages.

  If you encounter problems loading CUDA or cuDNN libraries, make sure your
  `LD_LIBRARY_PATH` does not contain paths to older CUDA/cuDNN libraries.

- _**GPU** support on macOS_

  The support for Apple Silicon GPUs in PyTorch+Keras is currently not great.
  Apple is working on `mlx` backend for Keras, which might improve the situation
  in the future.

  You can instead use the TensorFlow backend – just install `tensorflow~=2.16.1` and
  `tensorflow-metal` packages from PyPI, and run `export KERAS_BACKEND=tensorflow`
  in every terminal before running assignment scripts.

- _How to install TensorFlow_

  If you would like to install also TensorFlow, run `pip install tensorflow-cpu`
  for CPU-only support, and `pip install tensorflow[and-cuda]` for Linux/WSL2
  GPU support. However, the paths to the CUDA libraries seem not to be detected
  correctly, so I had to run
  ```sh
  export LD_LIBRARY_PATH=$(echo VENV_DIR/lib/python*/site-packages/nvidia/*/lib | tr " " ":")
  ```
  in the terminal for the GPU support to work.

### TOCEntry: MetaCentrum

- _How to apply for MetaCentrum account?_

  After reading the [Terms and conditions](https://docs.metacentrum.cz/access/terms/),
  you can [apply for an account here](https://docs.metacentrum.cz/access/account/).

  After your account is created, please make sure that the directories
  containing your solutions are always **private**.

- _How to activate Python 3.10 on MetaCentrum?_

  On Metacentrum, currently the newest available Python is 3.10, which you need
  to activate in every session by running the following command:
  ```
  module add python/python-3.10.4-intel-19.0.4-sc7snnf
  ```

- _How to install the required virtual environment on MetaCentrum?_

  To create a virtual environment, you first need to decide where it will
  reside. Either you can find a permanent storage, where you have large-enough
  [quota](https://docs.metacentrum.cz/data/quotas/), or you can [use scratch
  storage for a submitted job](https://docs.metacentrum.cz/computing/scratch-storages/).

  TL;DR:
  - Run an interactive CPU job, asking for 16GB scratch space:
    ```
    qsub -l select=1:ncpus=1:mem=8gb:scratch_local=16gb -I
    ```

  - In the job, use the allocated scratch space as the temporary directory:
    ```
    export TMPDIR=$SCRATCHDIR
    ```

  - You should clear the scratch space before you exit using the `clean_scratch`
    command. You can instruct the shell to call it automatically by running:
    ```
    trap 'clean_scratch' TERM EXIT
    ```

  - Finally, create the virtual environment and install PyTorch in it:
    ```
    module add python/python-3.10.4-intel-19.0.4-sc7snnf
    python3 -m venv CHOSEN_VENV_DIR
    CHOSEN_VENV_DIR/bin/pip install --no-cache-dir --upgrade pip setuptools
    CHOSEN_VENV_DIR/bin/pip install --no-cache-dir keras~=3.0.5 --extra-index-url=https://download.pytorch.org/whl/cu118 torch~=2.2.0 torchaudio~=2.2.0 torchvision~=0.17.0 torchmetrics~=1.3.1 flashlight-text~=0.0.3 tensorboard~=2.16.2 transformers~=4.37.2 gymnasium~=1.0.0a1 pygame~=2.5.2
    ```

- _How to run a GPU computation on MetaCentrum?_

  First, read the official MetaCentrum documentation:
  [Basic terms](https://docs.metacentrum.cz/computing/concepts/),
  [Run simple job](https://docs.metacentrum.cz/computing/run-basic-job/),
  [GPU computing](https://docs.metacentrum.cz/computing/gpu-comput/),
  [GPU clusters](https://docs.metacentrum.cz/computing/gpu-clusters/).

  TL;DR: To run an interactive GPU job with 1 CPU, 1 GPU, 8GB RAM, and 16GB scatch
  space, run:
  ```
  qsub -q gpu -l select=1:ncpus=1:ngpus=1:mem=8gb:scratch_local=16gb -I
  ```

  To run a script in a non-interactive way, replace the `-I` option with the script to be executed.

  If you want to run a CPU-only computation, remove the `-q gpu` and `ngpus=1:`
  from the above commands.

### TOCEntry: AIC

- _How to install required packages on [AIC](https://aic.ufal.mff.cuni.cz)?_

  The Python 3.11.7 is available `/opt/python/3.11.7/bin/python3`, so you should
  start by creating a virtual environment using
  ```
  /opt/python/3.11.7/bin/python3 -m venv VENV_DIR
  ```
  and then install the required packages in it using
  ```
  VENV_DIR/bin/pip install --no-cache-dir keras~=3.0.5 --extra-index-url=https://download.pytorch.org/whl/cu118 torch~=2.2.0 torchaudio~=2.2.0 torchvision~=0.17.0 torchmetrics~=1.3.1 flashlight-text~=0.0.3 tensorboard~=2.16.2 transformers~=4.37.2 gymnasium~=1.0.0a1 pygame~=2.5.2
  ```

- _How to run a GPU computation on AIC?_

  First, read the official AIC documentation:
  [Submitting CPU Jobs](https://aic.ufal.mff.cuni.cz/index.php/Submitting_CPU_Jobs),
  [Submitting GPU Jobs](https://aic.ufal.mff.cuni.cz/index.php/Submitting_GPU_Jobs).

  TL;DR: To run an interactive GPU job with 1 CPU, 1 GPU, and 16GB RAM, run:
  ```
  srun -p gpu -c1 -G1 --mem=16G --pty bash
  ```

  To run a shell script requiring a GPU in a non-interactive way, use
  ```
  sbatch -p gpu -c1 -G1 --mem=16G SCRIPT_PATH
  ```

  If you want to run a CPU-only computation, remove the `-p gpu` and `-G1`
  from the above commands.

### TOCEntry: Git

- _Is it possible to keep the solutions in a Git repository?_

  Definitely. Keeping the solutions in a branch of your repository,
  where you merge them with the course repository, is probably a good idea.
  However, please keep the cloned repository with your solutions **private**.

- _On GitHub, do not create a **public** fork with your solutions_

  If you keep your solutions in a GitHub repository, please do not create
  a clone of the repository by using the Fork button – this way, the cloned
  repository would be **public**.

  Of course, if you just want to create a pull request, GitHub requires a public
  fork and that is fine – just do not store your solutions in it.

- _How to clone the course repository?_

  To clone the course repository, run
  ```
  git clone https://github.com/ufal/npfl138
  ```
  This creates the repository in the `npfl138` subdirectory; if you want a different
  name, add it as a last parameter.

  To update the repository, run `git pull` inside the repository directory.

- _How to keep the course repository as a branch in your repository?_

  If you want to store the course repository just in a local branch of your
  existing repository, you can run the following command while in it:
  ```
  git remote add upstream https://github.com/ufal/npfl138
  git fetch upstream
  git checkout -t upstream/master
  ```
  This creates a branch `master`; if you want a different name, add
  `-b BRANCH_NAME` to the last command.

  In both cases, you can update your checkout by running `git pull` while in it.

- _How to merge the course repository with your modifications?_

  If you want to store your solutions in a branch merged with the course
  repository, you should start by
  ```
  git remote add upstream https://github.com/ufal/npfl138
  git pull upstream master
  ```
  which creates a branch `master`; if you want a different name,
  change the last argument to `master:BRANCH_NAME`.

  You can then commit to this branch and push it to your repository.

  To merge the current course repository with your branch, run
  ```
  git merge upstream master
  ```
  while in your branch. Of course, it might be necessary to resolve conflicts
  if both you and I modified the same place in the templates.

### TOCEntry: ReCodEx

- _What files can be submitted to ReCodEx?_

  You can submit multiple files of any type to ReCodEx. There is a limit of
  **20** files per submission, with a total size of **20MB**.

- _What file does ReCodEx execute and what arguments does it use?_

  Exactly one file with `py` suffix must contain a line starting with `def main(`.
  Such a file is imported by ReCodEx and the `main` method is executed
  (during the import, `__name__ == "__recodex__"`).

  The file must also export an argument parser called `parser`. ReCodEx uses its
  arguments and default values, but it overwrites some of the arguments
  depending on the test being executed – the template should always indicate which
  arguments are set by ReCodEx and which are left intact.

- _What are the time and memory limits?_

  The memory limit during evaluation is **1.5GB**. The time limit varies, but it should
  be at least 10 seconds and at least twice the running time of my solution.

### TOCEntry: Finetuning

- _How to make a part of the network frozen, so that its weights are not updated?_

  Each `keras.layers.Layer`/`keras.Model` has a mutable `trainable`
  property indicating whether its variables should be updated – however, after
  changing it, you need to call `.compile` again (or otherwise make sure the
  list of trainable variables for the optimizer is updated).

  Note that once `trainable == False`, the insides of a layer are no longer
  considered, even if some its sub-layers have `trainable == True`. Therefore, if
  you want to freeze only some sub-layers of a layer you use in your model, the
  layer itself must have `trainable == True`.

- _How to choose whether dropout/batch normalization is executed in training or
  inference regime?_

  When calling a `keras.layers.Layer`/`keras.Model`, a named option
  `training` can be specified, indicating whether training or inference regime
  should be used. For a model, this option is automatically passed to its layers
  which require it, and Keras automatically passes it during
  `model.{fit,evaluate,predict}`.

  However, you can manually pass for example `training=False` to a layer when
  using Functional API, meaning that layer is executed in the inference
  regime even when the whole model is training.

- _How does `trainable` and `training` interact?_

  The only layer, which is influenced by both these options, is batch
  normalization, for which:
  - if `trainable == False`, the layer is always executed in inference regime;
  - if `trainable == True`, the training/inference regime is chosen according
    to the `training` option.


### TOCEntry: TensorBoard

- _Cannot start TensorBoard after installation_

  If `tensorboard` executable cannot be found, make sure the directory with pip installed
  packages is in your PATH (that directory is either in your virtual environment
  if you use a virtual environment, or it should be `~/.local/bin` on Linux
  and `%UserProfile%\AppData\Roaming\Python\Python311` and
  `%UserProfile%\AppData\Roaming\Python\Python311\Scripts` on Windows).

- _What can be logged in TensorBoard?_
  See the documentation of the [`SummaryWriter`](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter).
  Common possibilities are:
  - scalar values:
    ```python
    summary_writer.add_scalar(name like "train/loss", value, step)
    ```
  - tensor values displayed as histograms or distributions:
    ```python
    summary_writer.add_histogram(name like "train/output_layer", tensor, step)
    ```
  - images as tensors with shape `[num_images, h, w, channels]`, where
    `channels` can be 1 (grayscale), 2 (grayscale + alpha), 3 (RGB), 4 (RGBA):
    ```python
    summary_writer.add_images(name like "train/samples", images, step, dataformats="NHWC")
    ```
    Other dataformats are `"HWC"` (shape `[h, w, channels]`), `"HW"`, `"NCHW"`, `"CHW"`.
  - possibly large amount of text (e.g., all hyperparameter values, sample
    translations in MT, …) in Markdown format:
    ```python
    summary_writer.add_text(name like "hyperparameters", markdown, step)
    ```
  - audio as tensors with shape `[1, samples]` and values in $[-1,1]$ range:
    ```python
    summary_writer.add_audio(name like "train/samples", clip, step, [sample_rate])
    ```
