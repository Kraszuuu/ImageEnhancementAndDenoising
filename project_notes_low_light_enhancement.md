# Project: Deep Learning-Based Low-Light Image Enhancement and Denoising

## 1. Project context

The project is intended for the courses **Digital Image Processing** and **Deep Learning**.

The selected research problem is low-light image enhancement combined with noise reduction. The project will investigate whether a deep learning-based approach can improve image quality while remaining robust to different image conditions.

## 2. Main aim

Develop and evaluate a deep learning-based method for low-light image enhancement and noise reduction.

The study will examine how combining real and synthetic image datasets with controlled noise augmentation affects image quality, model robustness and generalisation. The proposed method will also be compared with conventional image-processing techniques.

## 3. Data

Datasets considered:
- **LOL** — paired low-light and normal-light images;
- **LOL-v2 Real** — real-world low-light image pairs;
- **LOL-v2 Synthetic** — synthetically generated low-light image pairs.

Controlled noise may be added to images to study denoising performance and improve robustness. Candidate noise models include Gaussian and Poisson noise.

Use separate training, validation and test data. Test images must not be used during training or model selection.

## 4. Proposed method

- Use a CNN-based architecture, with **U-Net** as a practical starting point.
- Train or adapt the model for low-light enhancement and noise reduction.
- Use real and synthetic data, with controlled noise augmentation.
- Compare the deep learning method against traditional image-processing baselines.

Potential baselines:
- gamma correction;
- CLAHE (Contrast Limited Adaptive Histogram Equalization);
- Retinex-based enhancement;
- conventional denoising filters.

The final architecture and training setup are to be selected after inspecting the datasets and available computational resources.

## 5. Evaluation

Primary quantitative metrics:
- **PSNR** — reconstruction quality;
- **SSIM** — structural similarity.

Possible additional metrics:
- **LPIPS** — perceptual similarity;
- **NIQE** — no-reference image quality;
- inference time — computational efficiency.

Compare the proposed method with traditional enhancement methods. Where practical, evaluate the effect of noise augmentation and dataset composition. Visual comparisons should use the same test images and consistent display settings.

## 6. Tools and implementation

Proposed tools:
- Python;
- PyTorch (or TensorFlow);
- OpenCV;
- NumPy;
- scikit-image;
- Matplotlib;
- Jupyter Notebook.

A simple **Streamlit** interface may be added to demonstrate the method, but it is optional and secondary to the experiments.

## 7. Expected outcome

A working prototype that enhances low-light images and reduces noise, accompanied by a quantitative and visual comparison with conventional methods.

The project should also discuss limitations, including generalisation to unseen real-world images and the trade-off between enhancement quality and inference time.

## 8. Decisions and open points

### Agreed direction
- Focus on **low-light image enhancement and denoising** as one combined problem.
- Use paired low-light/normal-light datasets, prioritising LOL and LOL-v2.
- Include controlled noise augmentation as an experimental factor.
- Compare the proposed model with established image-processing baselines.
- Use PSNR and SSIM as the core evaluation metrics.

### Still to be finalised
- Exact dataset subset and train/validation/test split.
- Final model architecture and loss function.
- Noise types, levels and augmentation strategy.
- Whether to include LPIPS, NIQE and inference-time measurements.
- Whether to build the optional Streamlit demo.

## 9. Working project title

**Deep Learning-Based Low-Light Image Enhancement and Denoising**
