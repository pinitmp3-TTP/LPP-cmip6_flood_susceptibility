# Future flood susceptibility of the Lam Phra Phloeng watershed under screened CMIP6 scenarios

Code for the manuscript *"Future Flood Susceptibility of the Reservoir-Regulated Lam Phra Phloeng Watershed under Dynamically-Screened CMIP6 Scenarios: A Two-Stage Attribution of Climate and Land-Use Drivers"* (Tanachaichoksirikun, Seeboonruang et al.). Funded by TSRI Fundamental Fund, contract RE-KRIS/FF69/46.

The workflow has two stages. Stage I learns a present-day flood-susceptibility pattern S(x) from a multi-event Sentinel-1 SAR inventory and 11 static factors with an RF/SVM/ANN/XGBoost ensemble. Stage II scales the prevalence-recalibrated flood odds with a gauge-based climate factor λ derived from five dynamically-screened, EQM-corrected CMIP6 models (R95p ratio, 1985–2014 baseline; SSP1-2.6/2-4.5/5-8.5; 2036–2065 and 2071–2100). A bounded factorial separates climate and land-use effects, and a reservoir analysis splits the basin by the dam catchment.

## Environment

* **GEE** scripts (`*_GEE_*.js`) run in the Google Earth Engine Code Editor. Each one exports to the Google Drive folder `GEE_LPP_CMIP6`. After a run, open **Tasks** and press **RUN** for every export.
* **Python** scripts (`*.py`) are written for Google Colab. Paste one file into a cell and run it. Each script mounts Drive and reads and writes under `/content/drive/MyDrive/GEE_LPP_CMIP6`. Outside Colab they use the current directory.
* Python packages are listed in `requirements.txt`.
* The GEE asset IDs at the top of `01_GEE_stage1_predictor_stack_samples.js` (watershed boundary, SAR flood-frequency map) and in the other GEE scripts point to the authors' Earth Engine project. Replace them with your own copies.

## External inputs not produced by this code

| Input | Location in Drive | Notes |
|---|---|---|
| EQM-corrected daily rainfall, 22 CMIP6 models × 9 RID gauges | `GEE_LPP_CMIP6/03_QM_Corrected/` | Tree `Historical/pr/<MODEL>/`, `Future_NearTerm/<ssp>/<MODEL>/` (2015–2054), `Future_FarTerm/<ssp>/<MODEL>/` (2055–2100). Each CSV holds `Date, pr_gcm, pr_qm_corrected`. Produced by the GCM pre-selection study (Tanachaichoksirikun and Seeboonruang, under review). |
| Sentinel-1 flood-frequency map (0–3 events) | GEE asset | From the companion SAR inventory study (Tanachaichoksirikun et al., under review). |
| Watershed boundary | GEE asset | |

## Run order

Run the scripts in numerical order. Each one depends only on scripts with lower numbers.

| # | Script | Platform | What it does | Main outputs | Paper |
|---|---|---|---|---|---|
| 01 | `01_GEE_stage1_predictor_stack_samples.js` | GEE | 11-factor predictor stack (30 m) and balanced presence/absence sample (2,000 + 2,000, 150 m buffer) | `LPP_paper5_predictor_stack_static.tif`, `LPP_paper5_samples_static.csv` | 2.3, 3.2 |
| 02 | `02_GEE_export_flood_frequency.js` | GEE | Exports the SAR flood-frequency map on the same grid. **Move the file into `LULC/`.** | `LULC/LPP_flood_frequency.tif` | 2.2, 3.5 |
| 03 | `03_stage2_ml_ensemble.py` | Colab | VIF screening, RF/SVM/ANN/XGBoost, random and spatial-block CV, weighted and stacking ensembles, SHAP | `stage2_outputs/models/`, `metrics_cv.csv`, `shap_*.png` | 3.3–3.4, 4.1–4.2, Table 2, Fig 3 |
| 04 | `04_stage3_present_map.py` | Colab | Wall-to-wall present susceptibility, five equal-area classes, model-disagreement layer | `present_susceptibility*.tif`, `susceptibility_breaks.json` | 4.2, Fig 4 |
| 05 | `05_stage3b_validation.py` | Colab | Inventory by class, success-rate curve, frequency gradient | `validation_*.csv/png` | 4.3, Table 3, Fig 5 |
| 06 | `06_stageII_a_compute_indices.py` | Colab | R10mm and R95p (wet-day 95th percentile fixed on 1985–2014) per model, gauge and year | `indices_output/indices_annual_long.csv` | 2.4 |
| 07 | `07_stageII_b_lambda_basin_future.py` | Colab | Basin-mean λ per GCM (three in-basin gauges) and first-pass future maps | `stage2_outputs/futureII/lambda_table.csv` | 3.6 |
| 08 | `08_stageII_c_per_station_lambda.py` | Colab | λ per gauge for the six gauges (screened-5 mean, SD, number of models with λ > 1) | `per_station_lambda.csv` | 3.6, Table 5 |
| 09 | `09_GEE_station_zones.js` | GEE | Nearest-gauge (Voronoi) zones on the 30 m grid | `lambda_zones_station_id.tif` | 3.6 |
| 10 | `10_stageII_d_spatial_lambda_recalibration.py` | Colab | Prevalence recalibration (τ ≈ 0.023), λ(x) by zone, elasticity sweep η ∈ {0.5, 1, 2, 3}, four-of-five agreement map | `stage2_outputs/futureII_spatial/` | 3.4, 3.6, 4.5, Table 5 |
| 11 | `11_rq2_screening_test.py` | Colab | Screened 5 vs unscreened 17 vs all 22 models | `revision_outputs/rq2_screening_comparison.*` | 4.4, Table 4, Fig 6a |
| 12 | `12_rq2_all_5of22_subsets.py` | Colab | Screened-5 λ against all C(22,5) = 26,334 five-model subsets | `revision_outputs/rq2_bootstrap_5of22.*` | 4.4, Table 4, Fig 6b |
| 13 | `13_lambda_raw_vs_eqm_diagnostic.py` | Colab | GCM-cell membership of each gauge; λ from raw GCM vs after EQM | `revision_outputs/round2_lambda_diag/` | 2.4, 4.5 |
| 14 | `14_GEE_stage4a_export_dynamic_world.js` | GEE | Dynamic World 3-year modal composites (2016, 2020, 2024) remapped to WorldCover classes | `LULC/DW_lulc_*.tif` | 2.5 |
| 15 | `15_stage4b_ca_markov_2045.py` | Colab | Markov–CA land-cover projection to 2045 | `LULC/lulc_2045_camarkov.tif` | 2.5 |
| 16 | `16_lulc_hindcast_validation.py` | Colab | 2024 hindcast from 2016–2020 (overall accuracy, κ, Figure of Merit) | `LULC/hindcast_validation.csv` | 4.6 |
| 17 | `17_factorial_attribution.py` | Colab | Bounded 2×2 climate × land-use factorial on changed pixels, by transition, on both probability scales | `stage2_outputs/attribution_fair/` | 3.7, 4.7, Table 6 |
| 18 | `18_GEE_buffer_samples.js` | GEE | Samples with 0 m and 500 m absence buffers. **Run twice**: `BUFFER_M = 0`, then `BUFFER_M = 500` (line 9). | `LPP_paper5_samples_buffer0.csv`, `_buffer500.csv` | 4.2 |
| 19 | `19_cv_loro_buffer_autocorrelation.py` | Colab | Spatial autocorrelation range, leave-one-region-out CV, buffer sensitivity | `revision_outputs/loro_cv.csv`, `buffer_sensitivity.csv`, `autocorrelation_range.csv` | 4.2 |
| 20 | `20_cv_block_size_3_5_6km.py` | Colab | Spatial-block CV with 3, 5 and 6 km blocks for every learner and the ensemble | `revision_outputs/round2_block_cv_3km_vs_5km.csv` | 4.2, Table 2 |
| 21 | `21_GEE_reservoir_zones_population_flowdir.js` | GEE | Reservoir pool, seasonal drawdown ring, WorldPop 2020, MERIT Hydro D8 flow direction (native grid). **Run all three exports.** | `reservoir_zone.tif`, `population.tif`, `merit_flowdir.tif` | 3.8 |
| 22 | `22_dam_catchment_updown.py` | Colab | Dam catchment from flow direction; rebuilds the upstream/downstream zones | `reservoir_zone_hydro.tif`, `revision_outputs/round2_dam_catchment.csv` | 3.8, 4.8 |
| 23 | `23_reservoir_exposure_summary.py` | Colab | Zone statistics (susceptibility, inventory density, capture) and sum-preserving cropland and population exposure | `stage2_outputs/revision_reservoir_exposure/` | 4.8 |
| 24 | `24_drawdown_sensitivity.py` | Colab | Re-fits the ensemble without drawdown-ring samples and compares skill and predictions | `revision_outputs/round2_drawdown_sensitivity.csv` | 4.8 |
| 25 | `25_consistency_checks_lambda_builtup.py` | Colab | Zone-area-weighted λ; observed vs projected built-up growth | `revision_outputs/round2_area_weighted_lambda.csv`, `round2_builtup_growth_check.csv` | 4.5, 4.6, Table 5 |
| 26 | `26_round3_uniform_lambda_ring_mask.py` | Colab | Area change with λ(x) vs a basin-uniform λ, with the reservoir pool and drawdown ring removed; elasticity sweep; exposure without the ring; cropland→water flicker check | `revision_outputs/round3_*.csv` | 4.5, 4.7, 4.8, Table 5 |
| 27 | `27_fig2_workflow_fig3_shap.py` | Colab | Workflow diagram; SHAP re-plot of the saved XGBoost model with readable class names | `revision_outputs/fig2_workflow.png`, `fig3_shap.png` | Fig 2, Fig 3 |
| 28 | `28_fig4_fig7_maps.py` | Colab | Present susceptibility map and far-future SSP5-8.5 change / model-agreement maps, reservoir pool and drawdown ring hatched, discrete agreement scale | `revision_outputs/fig4_present.png`, `fig7_robustness.png` | Fig 4, Fig 7 |
| 29 | `29_fig8_attribution_bar.py` | Colab | Factorial attribution bar chart (recalibrated scale when 17 has written it) | `revision_outputs/fig8_attribution_bar.png` | Fig 8 |

Figure 1 (study area) was drawn in GIS and is not produced by this code.

## Notes on reproducibility

* All learners and samplers use seed 42 (`SEED` in 01, 03 and 18). Script 20 re-fits the saved models with `sklearn.clone`, so its 3 km values can differ slightly (third decimal) from Table 2; the 3/5/6 km comparison is made within one run.
* Two probability scales are used, as in the paper. The **balanced-sample** scale is the raw ensemble output (03, 04, 17). The **prevalence-recalibrated** scale is produced by 10 and used by 10, 17 (second table), 22–26 and 28–29.
* In 13, future rainfall files are concatenated across folder terms and cut to 2036–2065 and 2071–2100. Its self-check prints the in-basin far-future SSP5-8.5 λ, which should equal Table 4 (1.116).
* Script 22 uses `pysheds` (installed automatically in Colab). It delineates a dam catchment of 704 km². The value reported by Sirikaew et al. (2020) is 815.2 km².

## Citation

Please cite the article (details on publication) and this archive (Zenodo DOI on publication).
