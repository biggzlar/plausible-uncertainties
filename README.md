# Plausible Uncertainties
This repo contains methods used in the paper *["Plausible Uncertainties for Human Pose Regression"](https://openaccess.thecvf.com/content/ICCV2023/papers/Bramlage_Plausible_Uncertainties_for_Human_Pose_Regression_ICCV_2023_paper.pdf)* published at ICCV 2023. Where noted in the code, these methods have been adapted from prior work. They also include our own interpretation of multivariate deep evidential regression.

For further information, see the following links: <br>
[Supplementary](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Bramlage_Plausible_Uncertainties_for_ICCV_2023_supplemental.pdf) <br>
[Intro video](https://youtu.be/mMEeU1Zm3iY)

# Examples
## Univariate
Deep evidential regression                                              |  Kendall-style MLE and MC dropout
:----------------------------------------------------------------------:|:----------------------------------------------------------------------:
![Deep evidential regression](images/UnivariateDerNet_performance.svg)  |  ![Kendall-style MLE and MC dropout](images/UnivariateKenNet_performance.svg)

## Multivariate
Deep evidential regression                                              |  Kendall-style MLE and MC dropout
:----------------------------------------------------------------------:|:----------------------------------------------------------------------:
![Deep evidential regression](images/MultivariateDerNet_performance.svg)  |  ![Kendall-style MLE and MC dropout](images/MultivariateKenNet_performance.svg)
