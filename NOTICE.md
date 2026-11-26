# Sources and licenses

The MIT license applies to code written for this project. It does not
relicense third-party dependencies or their models. EEGConformer, EEGNeX,
ATCNet and FBCNet are imported from Braindecode 1.2.0 at runtime; its source
is not bundled here.

EEGNet, Lee-CRNN, PSD-AE-RBFNN and TDCNet use study adaptations based on
their published methods:

- EEGNet: [Lawhern et al.](https://doi.org/10.1088/1741-2552/aace8c)
- Lee-CRNN: [Lee et al.](https://doi.org/10.1016/j.jneumeth.2021.109282)
- PSD-AE-RBFNN: [Jibon et al.](https://doi.org/10.1177/20552076241297355)
- TDCNet: [Xu et al.](https://doi.org/10.1142/S0129065726500486); no
  upstream repository source is distributed.

The [ds002778](https://doi.org/10.18112/openneuro.ds002778.v1.0.5) and
[ds003490](https://doi.org/10.18112/openneuro.ds003490.v1.1.0) metadata
declare CC0. Dataset citations remain appropriate for derived input files.
No model checkpoint or participant-level prediction is stored in Git.
