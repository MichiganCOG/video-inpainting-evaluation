# Video Inpainting Evaluation

This project evaluates metrics used to evaluate video inpainting models. It includes metrics that measure reconstruction
performance (PSNR, SSIM, LPIPS, PVCS), realism (FID, video FID), and temporal consistency (warping error, patch
consistency).

Note: This project is in maintenance mode due to the author graduating. Errors will be addressed to the greatest extent possible, but new features and upgrades will not.

## Installation and Setup

1. Create an environment and install project dependencies with conda:

    ```bash
    conda env create -p ./env -f environment.yml
    ```

   Always make sure this environment is activated when running the commands in this document, and that commands are run
   from this project's root directory unless otherwise specified.

   ```bash
   conda activate ./env
   ```

2. Run `install-flownet2.sh` to compile FlowNet2's custom layers:

    ```bash
    ./scripts/setup/install-flownet2.sh
    ```

3. Download model weights (e.g., for flow computation, (V)FID feature extraction, etc.):

    ```bash
    ./scripts/setup/download-models.sh
    ```

## Data Preparation

This project expects a two-tier file structure for "ground-truth" frames (i.e., normal video frames), masks, and 
inpainted frames. Ground-truth and mask frames should go in the same folder and follow the structure shown below:

```text
frames/
├── video1/
│   ├── frame_0000_gt.png
│   ├── frame_0000_mask.png
│   ├── frame_0001_gt.png
│   ├── frame_0001_mask.png
│   ├── frame_0002_gt.png
│   ├── frame_0002_mask.png
│   └── ...
├── video2/
│   ├── frame_0000_gt.png
│   ├── frame_0000_mask.png
│   ├── frame_0001_gt.png
│   ├── frame_0001_mask.png
│   ├── frame_0002_gt.png
│   ├── frame_0002_mask.png
│   └── ...
...
```

Inpainted results should go in a different folder and follow the structure shown below:

```text
inpainting-results/
├── video1/
│   ├── frame_0000_pred.png
│   ├── frame_0001_pred.png
│   ├── frame_0002_pred.png
│   └── ...
├── video2/
│   ├── frame_0000_pred.png
│   ├── frame_0001_pred.png
│   ├── frame_0002_pred.png
│   └── ...
...
```

### Computing Evaluation Features

Various features (e.g., flow, (V)FID features, etc.) must be pre-computed before running the evaluation script. To
produce these features, run `compute-evaluation-features.sh`. The example below consumes a dataset stored in the
`frames` directory and saves the features in the `eval-data` directory:

```bash
./scripts/preprocessing/compute-evaluation-features.sh frames eval-data
```

## Evaluation

Run `evaluate_inpainting.py`:

```bash
python -m src.main.evaluate_inpainting \
    --gt_root=frames \
    --pred_root=inpainting-results \
    --eval_feats_root=eval-data \
    --output_path=quantitative-results.npy
```

Specific metrics can be included or excluded with the `--include` and `--exclude` flags respectively (supported keys are
listed in the [Metrics section](#metrics)). For more options, call `python -m src.main.evaluate_inpainting -h`.

## Metrics

The table below lists the supported metrics with brief descriptions.

<table>
  <tr>
    <th>Metric</th>
    <th>Key</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Peak Signal-to-Noise Ratio</td>
    <td><code>psnr</code></td>
    <td>
      Computes the Peak Signal-to-Noise Ratio (PSNR) between each inpainted frame and its corresponding ground-truth
      frame.
    </td>
  </tr>
  <tr>
    <td>Structural Similarity</td>
    <td><code>ssim</code></td>
    <td>
      Computes the Structural Similarity (SSIM) between each inpainted frame and its corresponding ground-truth frame.
    </td>
  </tr>
  <tr>
    <td>Learned Perceptual Image Patch Similarity</td>
    <td><code>lpips</code></td>
    <td>
      Computes the average Learned Perceptual Image Patch Similarity (LPIPS)<sup>1</sup> between each inpainted frame
      and its corresponding ground-truth frame. We use a VGG model fine-tuned for the patch similarity task.
    </td>
  </tr>
  <tr>
    <td>Perceptual Video Clip Similarity</td>
    <td><code>pvcs</code></td>
    <td>
      Computes the average Perceptual Video Clip Similarity (PVCS) between each 10-frame clip of an inpainted video and
      the corresponding 10-frame clip from the ground-truth video. For each pair of clips, we extract the activations of
      the <code>Conv3d_2c_3x3</code>, <code>Mixed_3c</code>, <code>Mixed_4f</code>, and <code>Mixed_5c</code> layers of
      a pretrained I3D backbone<sup>2</sup>, and compute the distances between features using the spatial feature
      distance function from LPIPS<sup>1</sup>. 
    </td>
  </tr>
  <tr>
    <td>Frechet Inception Distance</td>
    <td><code>fid</code></td>
    <td>
      Computes the Frechet Inception Distance (FID)<sup>3</sup> between the distribution of all inpainted frames and the
      distribution of all corresponding ground-truth frames.
    </td>
  </tr>
  <tr>
    <td>Video Frechet Inception Distance</td>
    <td><code>vfid</code></td>
    <td>
      Computes the Video Frechet Inception Distance (VFID) between the distribution of inpainted videos and the
      distribution of corresponding ground-truth videos. We use the logits from a pretrained I3D backbone<sup>2</sup> to
      represent each video.
    </td>
  </tr>
  <tr>
    <td>Video Frechet Inception Distance on Clips</td>
    <td><code>vfid_clips</code></td>
    <td>
      Computes the Video Frechet Inception Distance (VFID) between the distribution of 10-frame clips from inpainted
      videos and the distribution of 10-frame clips from corresponding ground-truth videos. This is similar to Video 
      Frechet Inception Distance (VFID), except 10-frame clips are used instead of entire videos to increase sample 
      size.
    </td>
  </tr>
  <tr>
    <td>Patch Consistency (PSNR)</td>
    <td><code>pcons_psnr</code></td>
    <td>
      Computes the average patch consistency<sup>4</sup> between all consecutive pairs of inpainted frames. For each 
      pair, a random patch from the first frame is sampled and compared to the neighborhood of patches in the next 
      frame. The maximum PSNR between the first-frame patch and second-frame patches is defined as the patch consistency
      between the pair.
    </td>
  </tr>
  <tr>
    <td>Patch Consistency (SSIM)</td>
    <td><code>pcons_ssim</code></td>
    <td>
      Computes the average patch consistency<sup>4</sup> between all consecutive pairs of inpainted frames. It is the
      same as Patch Consistency (PSNR), except it uses SSIM in place of PSNR.
    </td>
  </tr>
  <tr>
    <td>Masked Patch Consistency (PSNR)</td>
    <td><code>pcons_psnr_mask</code></td>
    <td>
      Computes the average masked patch consistency between all consecutive pairs of inpainted frames. It is the same as 
      Patch Consistency (PSNR), except the sampled patch is always centered at the centroid of the inpainted region.
    </td>
  </tr>
  <tr>
    <td>Warping Error</td>
    <td><code>warp_error</code></td>
    <td>
      Computes the average warping error<sup>5</sup> across all pairs of consecutive inpainted frames.
    </td>
  </tr>
  <tr>
    <td>Masked Warping Error</td>
    <td><code>warp_error_mask</code></td>
    <td>
      Computes the average masked warping error across all pairs of consecutive inpainted frames. This is similar to
      Warping Error, except it is only computed over optical flow vectors that begin or end inside an inpainted region.
    </td>
  </tr>
</table>

### References

1. [Zhang et al. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR 2018.](
https://arxiv.org/abs/1801.03924)
2. [Carreira and Zisserman. Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. CVPR 2017.](
https://arxiv.org/abs/1705.07750)
3. [Heusel et al. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. NeurIPS 2017.](
https://arxiv.org/abs/1706.08500)
4. [Gupta et al. Characterizing and Improving Stability in Neural Style Transfer. ICCV 2017.](
https://arxiv.org/abs/1705.02092)
5. [Lai et al. Learning Blind Video Temporal Consistency. ECCV 2018.](https://arxiv.org/abs/1808.00449)

## Licenses

This software uses code from other projects under various licenses:

| Path                   | Source                                                            | License      |
|------------------------|-------------------------------------------------------------------|--------------|
| `/src/models/i3d`      | https://github.com/piergiaj/pytorch-i3d                           | Apache 2.0   |
| `/src/lpips`           | https://github.com/richzhang/PerceptualSimilarity                 | BSD 2-Clause |
| `/src/fid/pytorch_fid` | https://github.com/mseitzer/pytorch-fid                           | Apache 2.0   |
| `/src/models/flownet2` | https://github.com/andrewjong/flownet2-pytorch-1.0.1-with-CUDA-10 | Apache 2.0   |

Exact licenses for the above libraries are available in their respective paths.

Additionally, code attributed to `https://github.com/phoenix104104/fast_blind_video_consistency` is available under the
following MIT license:

```text
MIT License

Copyright (c) 2018 UC Merced Vision and Learning Lab
Modifications copyright (c) 2021 Ryan Szeto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

All other code, unless otherwise noted, is available under the MIT license in `LICENSE`.
