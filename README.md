# dssiam
This is a compacted version of the original code corresponding with the paper "Domain Adaptation for Single Object Tracking", which can be found [here](https://scripties.uba.uva.nl/search?id=712690;setlang=en) See abstract below:

<em>Visual object tracking has quickly become one of the major challenges within the domain of computer vision. However, the limited availability of annotated training data introduces a bias towards existing datasets in tracker development. In this work, we introduce a novel training method and regularization procedure for Siamese trackers that tackles this overfitting problem through offline deep supervision and divergence-based domain adaptation. The fully offline nature of our method harbors a significant advantage over previous external regularizers since it retains the original tracking speed of the underlying model. Our experiments on two renowned tracking benchmarks show promising results. When applied to a different target domain, the baseline
showed signs of severe overfitting on the source. We show that our deeply supervised training formulation reduces overfitting and uses data more efficiently compared to the baseline. Our method achieved a relative gain of 9% over the baseline on the most common accuracy and robustness metrics when using the same amount of training data. Additional divergence based regularization yields an even greater improvement in tracking robustness when object diversity is high, at the cost of localization.</em>
