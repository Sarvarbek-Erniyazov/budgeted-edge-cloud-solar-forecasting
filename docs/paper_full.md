# Full paper text (extracted)

Extracted read-only from `ICONI2026_Erniyazov_Lim_full.docx` with python-docx: body paragraphs, tables and captions in document order; page headers and footers are not included. `[image]` marks where a picture sits. Style names other than Normal are kept as HTML comments. Reference numbers [n] are Word auto-numbering, restored here from list order.

---

# When Does the Ground Sensor Need the Satellite? Budgeted Edge-to-Cloud Escalation for Solar Irradiance Forecasting

Sarvarbek Erniyazov and Chang Gyoon Lim

Department of Computer Engineering, Chonnam National University  <!-- style: Affiliation -->

Yeosu, South Korea  <!-- style: Affiliation -->

[e-mail: erniyazov_sarvarbek@jnu.ac.kr, cglim@jnu.ac.kr]  <!-- style: Affiliation -->

*Corresponding author: [TO CONFIRM]  <!-- style: Affiliation -->

[image]
Abstract

Short-term solar irradiance forecasting can combine ground measurements with satellite imagery and numerical weather prediction (NWP), but such multimodal models are typically run in the cloud, while a low-power device at a photovoltaic site sees only its own measurements. We ask how often such a device needs the cloud for intra-day global horizontal irradiance forecasting on the public Folsom benchmark. A compact on-device network using ground-sensor history is compared with a cloud forecaster that adds GOES-15 satellite patches and NWP forecasts available at issue time, and an on-device gate decides when to escalate a request. All settings and five claims were fixed on 2014-2015 data before a single evaluation on the held-out year 2016. There, the cloud forecaster lowers RMSE by 7.9% relative to the on-device network (70.66 versus 76.71 W/m2), and a day-block bootstrap interval excludes zero for all five seeds. With the model class held fixed, the cloud-side inputs lower RMSE by 4.2%. A size-capped ground-only tree model (87 kB) is 4.5% better than the on-device network but 3.7% worse than the cloud forecaster. At a 25% escalation target, an uncertainty-based gate retains 69% of the cloud gain while escalating 23% of requests, whereas random escalation of 25% retains 24%; a learned gate is no better. Of the five claims, two were refuted and one partly: we had expected the cloud-side inputs to add nothing and the size-capped tree model to match the cloud forecaster, and neither held; an expected absence of a cloud advantage on ramp events was also not confirmed. The study is a single-site measurement, and latency was measured on a PC CPU only.  <!-- style: Abstract Text -->

[image]
Keywords: Edge AI, on-device inference, edge-cloud collaboration, selective offloading, solar irradiance forecasting, satellite imagery  <!-- style: Abstract Text -->

## 1. Introduction

Intra-day solar irradiance forecasts support the scheduling and control of photovoltaic (PV) plants [1], [2]. The most accurate forecasts at these horizons combine ground measurements with satellite imagery and numerical weather prediction (NWP) [3], [4], and recent multimodal models fuse these sources with graph and attention-based architectures [5], [6]. Such models need data feeds and computation that are available in the cloud but not on a low-power device at the PV site, which sees only its own sensors.  <!-- style: p1a -->

A common answer in edge computing is to run a small model on the device and send only some requests to a larger model in the cloud [7]–[9], and this idea has recently been applied to PV power forecasting [10]. Microcontroller-class networks [11] and 8-bit integer inference [12] keep the on-device part small. Less is known about how much the cloud actually adds for solar forecasting, how often a device must ask for it, and whether a simple rule is enough to decide when.  <!-- style: p1a -->

This paper measures these quantities on the public Folsom benchmark [13]. We compare a 43.8 kB on-device network that uses ground-sensor history with a cloud forecaster that adds a GOES-15 satellite patch and NWP forecasts, and test five gates and an oracle that escalate a set share of requests. All settings and five claims were fixed on 2014-2015 data before a single evaluation on 2016. We separate the effect of the cloud-side inputs from that of the model class, measure how much of the cloud gain each gate keeps at each budget, and report the claims that failed as well as those that held. Code, the frozen protocol and all results are public at https://github.com/Sarvarbek-Erniyazov/budgeted-edge-cloud-solar-forecasting.  <!-- style: p1a -->

## 2. Related Work

Solar forecasting. Reviews describe how methods depend on the horizon and on the available inputs [1], [2]. Operational satellite and NWP forecasts have been validated against ground data across the United States [3], and deep-learning models now use sky and satellite images directly [4]. Our earlier work fused several data modalities with dynamic graph neural networks and a mixture of experts [5], and combined graph-enhanced patch segmentation with frequency fusion [6]. Both aim at accuracy in the cloud; here we ask how much of such a gain a device at the site can obtain by calling the cloud selectively.  <!-- style: p1a -->

Edge-cloud inference. Neurosurgeon partitions a network between device and cloud by layer [7], and distributed networks let confidently classified inputs exit early on the device [8], [9]; Qiao et al. combine cloud-side large models with edge models for PV power forecasting [10]. In our setting the tiers are separate models with different inputs, the share of escalated requests is set by a budget, and each gate is compared with random escalation at the same rate. MCUNet [11] and 8-bit integer inference [12] show how small on-device models can be; we apply the same int8 conversion and measure its cost.  <!-- style: p1a -->

Benchmark. The Folsom dataset [13] provides 2014-2016 ground, weather, sky-image, GOES-15 and NAM data with baselines, which makes protocol-level comparisons repeatable.  <!-- style: p1a -->

## 3. Problem and Protocol

Task. We forecast global horizontal irradiance at the Folsom site [13] for six horizons, 30 to 180 min in 30-min steps. Forecasts are issued every 30 min during daytime hours, and a cell (issue time, horizon) is scored when the solar elevation at the target time is at least 5°; the test year has 39,655 scored cells. Every input must be available at issue time: the satellite frame is used with an assumed 15-min availability lag, and NWP values come from the newest NAM run assumed available at issue time, 6 h after its reference time. Smart persistence is the reference forecast.  <!-- style: p1a -->

Splits. Models are trained on 2014. Gates are fitted on the first half of 2015 and validated on the second half; for the test, gates and their thresholds were refitted on all of 2015. The year 2016 is the test year.  <!-- style: p1a -->

Lock and single run. The data loader refuses 2016 rows unless the protocol is marked frozen and an explicit flag is set, which unit tests check. The protocol, its five claims and rules, and the code that evaluates them were committed and tagged before the test year was loaded, and the test was run once; in line with [14], everything is public.  <!-- style: p1a -->

Claims and rules. Table 1 lists the five claims. Each sub-test has a frozen rule for when it counts against its claim: claim 1, if the input effect is at least +2% with its interval above zero; claim 2 holds only if the capped trees are within 2% of the cloud tier and the interval is not above zero for three or more seeds; claim 3, if the interval includes zero for three or more seeds; claim 4, if the uncertainty gate keeps less than 0.10 more of the gain than random at 25% for three or more seeds, if the learned gate is better by 0.10 or more at two of three budgets, or if a realised rate misses its target by more than 0.05; claim 5, if the cloud tier has a ramp advantage for three or more seeds, or if a gate keeps half of the ramp gain at 25%. A claim is called supported if no sub-test counts against it, refuted if all do, and partly supported otherwise; this naming was applied after the run. On validation data the input effect had been −0.27% and the cloud gain 6.3%.  <!-- style: p1a -->

Table 1. Pre-specified claims and their 2016 outcome  <!-- style: p1a -->

| Claim | 2016 result | Verdict |
|---|---|---|
| 1. Cloud-side inputs add nothing (model class fixed) | +4.2%; interval 0.7 to 5.6 W/m2 | Refuted |
| 2. Size-capped ground-only trees match the cloud tier | 3.7% worse (87.2 kB model); 4 of 5 seeds | Refuted |
| 3. Cloud tier beats edge tier | 7.9% better; 5 of 5 seeds | Supported |
| 4. Gate beats random; learned gate not better; rates on budget | +0.38 to +0.52 share kept; max rate gap 0.023 | Supported |
| 5. No cloud advantage on ramps; no gate helps | advantage in 4 of 5 seeds; best gate 0.48 | Partly |

## 4. Method

Fig. 1 shows the two tiers and the gate. Both tiers forecast all six horizons at once.  <!-- style: p1a -->

[INSERT architecture.png HERE]  <!-- style: p1a -->

Fig. 1. System overview: edge tier and int8 gate on the device, cloud tier with satellite and NAM inputs.  <!-- style: p1a -->

## 4.1 Edge tier

The edge tier is a multilayer perceptron with layers 96-64-64-6 (10,758 parameters), one output per horizon. Its 96 inputs use only data up to the issue time: summary statistics of GHI and DNI over the day so far and over the last hour at 5-min resolution (72 features), means of the site weather sensors over the last 30 min (7) with a missing-data flag, solar elevation and clear-sky GHI at each target time (12), and the sine and cosine of day of year and hour of day (4). Inputs are standardised with training-set statistics. The network is trained with five seeds and exported to ONNX, at 43.8 kB in 32-bit floating point and 15.2 kB with 8-bit integer weights. It predicts the clear-sky index (clipped to [0, 1.2]) with ReLU hidden layers, mean squared error, AdamW and early stopping on the last 30 days of 2014; the int8 version uses static quantisation in ONNX Runtime.  <!-- style: p1a -->

## 4.2 Cloud tier

The cloud tier averages, with equal weights and in clear-sky index, two forecasters that receive the edge inputs plus a GOES-15 satellite tile around the site and NAM forecasts at four nearby grid nodes: histogram-based gradient-boosted trees, as in LightGBM [15] and implemented in scikit-learn [16], and a neural network in which two 3x3 convolution layers (16 and 32 channels) encode the 10x10 satellite tile and an MLP encodes the other inputs, trained like the edge tier with the same five seeds. For the control experiment in Section 5.2, the same tree model is also trained on the edge inputs alone (ground-only trees), once without a size limit and once capped at ten times the estimated int8 size of the edge network (cap 112 kB; compiled model 87.2 kB).  <!-- style: p1a -->

## 4.3 Gates

For each issue time, a gate on the device decides whether to request the cloud forecast under a target budget b, the share of issue times escalated. Gates decide on the 7,457 test issue times with a satellite frame and at least one daylight horizon; the other 434 daytime issue times are excluded from the primary evaluation. Random escalation picks exactly a share b, fixed-interval escalation spreads requests evenly within each day, and the variability rule scores the 30-min variability of the measured clear-sky index. The uncertainty gate, a 102-32-1 network on the 96 edge inputs and the six edge forecasts, predicts the mean absolute edge error; the learned gate has the same form and predicts the reduction in squared error from escalating. Both are trained on all of 2015 for the test. Score-based gates escalate when the score reaches the (1−b) quantile of its 2015 scores, so realised rates can differ from b. The oracle escalates the issue times with the largest true benefit and serves only as an upper reference.  <!-- style: p1a -->

## 4.4 Footprint and latency

Sizes are model file sizes (ONNX files for the networks, a compiled C object for the trees; runtime libraries excluded). Latency is the median of 5,000 single-row calls after 200 warm-up calls with one thread, using ONNX Runtime 1.30.0 or C compiled at -O2, on a PC CPU ([CPU MODEL]); nothing ran on a microcontroller. The gate size and latency were measured on development gates of the same architecture.  <!-- style: p1a -->

Evaluation. RMSE is computed per horizon and averaged over the six horizons on the 39,655 primary test cells. Networks are trained with five seeds and tree models are deterministic. Intervals are paired day-block bootstrap intervals [17]: local days are resampled 2,000 times with the same weights for both forecasts, and the 95% percentile interval of the RMSE difference is taken, separately for each seed. "Share kept" is (RMSEedge - RMSEgate) / (RMSEedge - RMSEcloud), the fraction of the edge-to-cloud gain that a gate retains.  <!-- style: p1a -->

## 5. Results

## 5.1 Main results

Table 2 summarises the test year. The edge tier improves on smart persistence (76.71 versus 89.43 W/m2), and the cloud tier lowers RMSE by a further 7.9% relative to the edge tier, to 70.66 W/m2; the interval excludes zero for all five seeds (claim 3). Across seeds the edge RMSE ranges from 75.4 to 79.8 W/m2 and the cloud RMSE from 70.1 to 71.1 W/m2. Per horizon (post hoc), the cloud gain is largest at 30 min (10.1%; 56.25 versus 50.54 W/m2) and lies between 6.6% and 8.1% from 60 to 180 min.  <!-- style: p1a -->

Table 2. Results on the 2016 test year (RMSE in W/m2)  <!-- style: p1a -->

| Model | RMSE | Kept | Rate | kB |
|---|---|---|---|---|
| Smart persistence | 89.43 | – | – | – |
| Edge tier (fp32) | 76.71 | 0.00 | 0 | 43.8 |
| Ground trees, capped | 73.29 | 0.57 | – | 87.2 |
| Trees, all inputs | 70.90 | 0.96 | – | – |
| Cloud tier | 70.66 | 1.00 | 1 | – |
| Random, 25% | 75.25 | 0.24 | 0.250 | – |
| Uncertainty, 25% | 72.61 | 0.69 | 0.230 | +6.1 |
| Learned, 25% | 72.79 | 0.65 | 0.248 | +6.1 |

## 5.2 Inputs versus model class

Two comparisons separate the cloud-side inputs from the model class. With the tree model fixed, adding the satellite tile and NAM forecasts lowers RMSE from 73.98 to 70.90 W/m2 (4.2%; interval of the difference 0.7 to 5.6 W/m2), so claim 1 is refuted. With the inputs fixed to the ground sensors, the size-capped trees (87.2 kB) are 4.5% better than the edge network (73.29 versus 76.71 W/m2) but remain 3.7% worse than the cloud tier, with the interval above zero for four of five seeds, so claim 2 is refuted. At this site, part of the edge-to-cloud gap therefore comes from the model class and part from the cloud-side inputs. On 2015 validation data the input effect had been −0.27%; we do not know why it differs on 2016.  <!-- style: p1a -->

## 5.3 Escalation budget

Fig. 2 shows RMSE against the realised escalation rate. At a 25% target, the uncertainty gate keeps 69% of the cloud gain at a realised rate of 23.0%, the learned gate 65% at 24.8%, the variability rule 40%, random escalation 24% and fixed-interval escalation 23%. The uncertainty gate beats random at all three report budgets: the paired RMSE difference is −2.23, −2.64 and −2.50 W/m2 at the 10, 25 and 50% targets, with the interval excluding zero for all five seeds. Against the learned gate the differences are −0.27, −0.18 and −0.27 W/m2, and the interval includes zero for every seed except one at 50%. All realised rates are within 0.023 of their targets. The ranking stated in claim 4 had no decision rule; the observed order agrees with it.  <!-- style: p1a -->

[INSERT budget_sweep_compact.png HERE]  <!-- style: p1a -->

Fig. 2. RMSE against realised escalation rate (2016; bands: seed range over five seeds).  <!-- style: p1a -->

## 5.4 Where the gate escalates

Fig. 3 shows where the uncertainty gate spends a 25% budget. Escalation is concentrated in the mornings and middays of January to May and October, and is rare from July to September and in the late afternoon. A post hoc split by the measured clear-sky index at issue time gives the same picture: the gate escalates 56-69% of issue times with an index below 0.8 and about 10% of those above 0.8. The cloud tier is better than the edge tier in every bin, by about 4.3 W/m2 under clear skies and by up to 11.4 W/m2 for indices between 0.2 and 0.8, so the gate spends most of its budget where the gain is largest but leaves a smaller gain under clear skies unused. Gate scores for this split were recomputed from the saved test gates and reproduce the stored decisions exactly.  <!-- style: p1a -->

[INSERT routing_map.png HERE]  <!-- style: p1a -->

Fig. 3. Share of issue times escalated by the uncertainty gate (25% target, 2016, mean of five seeds); blank cells have no decided issue times.  <!-- style: p1a -->

## 5.5 Ramp events

Ramp cells are primary cells whose observed clear-sky index changes by at least 0.25 from the previous horizon, 30 min earlier (2,551 in 2016). On them the cloud tier has lower RMSE than the edge tier (140.1 versus 145.9 W/m2, seed means), with the interval above zero for four of five seeds, which counts against claim 5. The best gate keeps 0.48 of the ramp gain at 25% (uncertainty gate, seed mean), below the 0.5 threshold of the second sub-test. The rule did not say whether the threshold applies to the seed mean or to each seed; the code committed before the test used the seed mean and excluded the oracle. Per seed, the uncertainty gate reaches 0.5 in two of five seeds and the learned gate in one, so claim 5 is partly supported and close to its threshold.  <!-- style: p1a -->

## 5.6 Quantisation, latency and link outage

With 8-bit weights the edge network shrinks from 43.8 to 15.2 kB, and its RMSE rises from 76.71 to 77.21 W/m2 (seed range 75.8 to 81.0 W/m2), a loss of 8% of the edge-to-cloud gain. On the PC CPU, the median single-row latency is 7.5 µs for the fp32 edge network, 9.6 µs for the int8 version, 8.6 µs for the uncertainty gate, 8.5 µs for the learned gate and 3.5 µs for the compiled capped trees. On this CPU int8 inference is slower than fp32, and none of these timings can be transferred to a microcontroller.  <!-- style: p1a -->

If the cloud cannot be reached, a refused request falls back to the edge forecast while the gate decisions stay the same. With whole days offline at random with probability 0.25 (200 draws), the uncertainty gate at a 25% target keeps 0.514 of the cloud gain, against 0.688 with the link always available; independent refusals with the same probability give 0.512. This analysis was named in the plan before the test run, but its setup was fixed after it.  <!-- style: p1a -->

## 6. Discussion and Limitations

Claims that did not hold. The input effect changed from −0.27% on the 2015 validation half-year to 4.2% on 2016, and an exploratory split of 2016 did not explain why; one validation half-year was too short to judge the inputs, and only the frozen test showed this. The oracle curve in Fig. 2 shows that a gate that knew the benefit could keep far more of the cloud gain than the uncertainty gate does.  <!-- style: p1a -->

Limitations. The study uses one site and one test year, so neither the size of the cloud gain nor the ranking of the gates is shown to transfer. The satellite availability lag and the NAM availability rule are assumptions, not measured delays. Sizes and latencies were measured on a PC CPU, not on a microcontroller. The 434 daytime issue times without a satellite frame are outside the primary evaluation. The per-horizon, clear-sky and link-outage analyses are post hoc or set up after the test, and are descriptive only.  <!-- style: p1a -->

## 7. Conclusions

On the Folsom benchmark, a 43.8 kB on-device network with a 6.1 kB gate kept 69% of the gain of a satellite and NWP cloud forecaster while escalating 23% of requests; a learned gate was no better than the uncertainty gate. The cloud-side inputs helped on the test year although they had not on validation data, and size-capped ground-only trees were 4.5% better than the on-device network but did not reach the cloud tier. Other sites and years, measured hardware, and gates that predict the benefit of escalation more accurately are left for future work.  <!-- style: p1a -->

## References

[1] D. Yang, J. Kleissl, C. A. Gueymard et al., “History and trends in solar irradiance and PV power forecasting: A preliminary assessment and review using text mining,” Solar Energy, vol.168, pp.60-101, 2018.

[2] D. Yang, W. Wang, C. A. Gueymard et al., “A review of solar forecasting, its dependence on atmospheric sciences and implications for grid integration: Towards carbon neutrality,” Renewable and Sustainable Energy Reviews, vol.161, 112348, 2022.

[3] R. Perez, S. Kivalov, J. Schlemmer et al., “Validation of short and medium term operational solar radiation forecasts in the US,” Solar Energy, vol.84, no.12, pp.2161-2172, 2010.

[4] Q. Paletta, G. Terrén-Serrano, Y. Nie et al., “Advances in solar forecasting: Computer vision with deep learning,” Advances in Applied Energy, vol.11, 100150, 2023.

[5] S. Erniyazov, M. A. Jaleel, C. G. Lim, and S. B. Ha, “A Multimodal Fusion Framework for Solar Forecasting using Dynamic GNNs and Mixture of Experts,” KSII Transactions on Internet and Information Systems, vol.20, no.3, pp.1337-1360, 2026.

[6] S. Erniyazov and C. G. Lim, “GNN-enhanced temporal patch segmentation and frequency fusion model for robust solar energy production forecasting,” Energy Reports, vol.13, pp.4962-4984, 2025.

[7] Y. Kang, J. Hauswald, C. Gao et al., “Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge,” in Proc. of the 22nd International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), pp.615-629, 2017.

[8] S. Teerapittayanon, B. McDanel, and H. T. Kung, “Distributed Deep Neural Networks Over the Cloud, the Edge and End Devices,” in Proc. of the IEEE 37th International Conference on Distributed Computing Systems (ICDCS), pp.328-339, 2017.

[9] S. Teerapittayanon, B. McDanel, and H. T. Kung, “BranchyNet: Fast inference via early exiting from deep neural networks,” in Proc. of the 23rd International Conference on Pattern Recognition (ICPR), pp.2464-2469, 2016.

[10] N. Qiao, S. Wang, S. Duan et al., “Cloud-Edge Collaborative Large Models for Robust Photovoltaic Power Forecasting,” arXiv preprint arXiv:2603.22343, 2026.

[11] J. Lin, W.-M. Chen, Y. Lin et al., “MCUNet: Tiny Deep Learning on IoT Devices,” in Proc. of Advances in Neural Information Processing Systems (NeurIPS), vol.33, 2020.

[12] B. Jacob, S. Kligys, B. Chen et al., “Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference,” in Proc. of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp.2704-2713, 2018.

[13] H. T. C. Pedro, D. P. Larson, and C. F. M. Coimbra, “A comprehensive dataset for the accelerated development and benchmarking of solar forecasting methods,” Journal of Renewable and Sustainable Energy, vol.11, no.3, 036102, 2019.

[14] D. Yang, “A guideline to solar forecasting research practice: Reproducible, operational, probabilistic or physically-based, ensemble, and skill (ROPES),” Journal of Renewable and Sustainable Energy, vol.11, no.2, 022701, 2019.

[15] G. Ke, Q. Meng, T. Finley et al., “LightGBM: A Highly Efficient Gradient Boosting Decision Tree,” in Proc. of Advances in Neural Information Processing Systems (NIPS), vol.30, 2017.

[16] F. Pedregosa, G. Varoquaux, A. Gramfort et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol.12, pp.2825-2830, 2011.

[17] H. R. Künsch, “The Jackknife and the Bootstrap for General Stationary Observations,” The Annals of Statistics, vol.17, no.3, pp.1217-1241, 1989.

