---
title: History
---

# Unreleased
- Added: NGA-East for USGS (2017) model (`NgaEastUsgs2017`), following the USGS nshmp-haz
  implementation: 17 weighted table-based median models for hard rock, the Stewart et al. (2017) and
  Hashash et al. (2017) site amplification, and a 0.2 panel / 0.8 EPRI standard deviation. The
  median and standard deviation are collapsed as in nshmp-haz, and reproduce its reference results.
  The seed-model logic tree and the Gulf Coastal Plain variants are not included.
- Added: NGA-Subduction USGS (2018) model (`NgaSubductionUsgs2018`), the USGS nshmp-haz version of
  Abrahamson et al. (2018, PEER 2018/02) with Cascadia adjustments for interface and intraslab events
  (`event_type`). `epistemic=False` gives the central branch (the nshmp-haz `*_NO_EPI` variants).
- Changed: Idriss (2014) accepts arrays of scenario values (`mag`, `dist_rup`, `v_s30`, `mechanism`).
  Arrays of N scenarios give results with shape (N, periods), e.g., `pga` with shape (N,).
  Scalar scenarios give the same results as before.
- Changed: `NumericParameter` and `CategoricalParameter` check arrays of values, with one warning
  for the values outside of the limits or options.
- Added: `ln_pga` property, the natural logarithm of PGA, which avoids computing `np.log(pga)`.
- Changed: Boore, Stewart, Seyhan, and Atkinson (2014) accepts arrays of scenario values (`mag`,
  `dist_jb`, `v_s30`, `depth_1_0`, `mechanism`, `region`) and the `ims` argument. Scalar scenarios give
  the same results as before.
- Changed: Abrahamson, Silva, and Kamai (2014), Campbell and Bozorgnia (2014), and Chiou and Youngs
  (2014) accept arrays of scenario values and the `ims` argument. Values that are estimated when not
  provided (e.g., `depth_tor`, `width`, `depth_1_0`, `depth_2_5`, `depth_hyp`) are estimated for each
  scenario. Scalar scenarios give the same results as before.
- Changed: Abrahamson, Gregor, and Addo (2016) and Coppersmith and Bommer (2014) accept arrays of
  scenario values (`mag`, `dist_rup`, `dist_hyp`, `depth_hyp`, `v_s30`, `event_type`,
  `tectonic_region`) and the `ims` argument, which follows the other keyword arguments. `dist_hyp`
  and `depth_hyp` may be *None* when no events are intraslab, and a missing distance or depth that is
  needed raises a `ValueError`.
- Changed: Akkar, Sandikkaya, and Bommer (2014), Derras, Bard, and Cotton (2014), and Hermkes, Kuehn,
  and Riggelsen (2014) accept arrays of scenario values and the `ims` argument. Derras et al. (2014)
  raises a `ValueError`, instead of a `KeyError`, for an unsupported mechanism.
- Changed: Atkinson and Boore (2006), Campbell (2003), Pezeshk et al. (2011), and Tavakoli and
  Pezeshk (2005) accept arrays of `mag` and `dist_rup` (and `v_s30` for Atkinson and Boore (2006))
  and the `ims` argument.
- Changed: `calc_width`, `calc_depth_tor`, `calc_depth_1_0`, and `calc_site_term` of Abrahamson et al.
  (2014); `calc_depth_2_5`, `calc_width`, `calc_depth_hyp`, `calc_depth_bor`, and `calc_site_term` of
  Campbell and Bozorgnia (2014); and `calc_depth_tor` and `calc_site_term` of Chiou and Youngs (2014)
  accept arrays.
- Changed: the duration models of Abrahamson and Silva (1996), Kempton and Stewart (2006), Afshari
  and Stewart (2016), and Pinilla-Ramos et al. (2023, 2024) accept arrays of scenario values. Arrays
  of N scenarios give durations and standard errors with shape (N,) (fields of the record arrays for
  Kempton and Stewart (2006) and Afshari and Stewart (2016)), and `AbrahamsonSilva1996.interp`
  returns shape (N, len(nias)), or (N, len(stds), len(nias)) with `stds`. Scalar scenarios give the
  same results as before. `AfshariStewart2016.calc_depth_1_0` accepts arrays of `v_s30` and
  `region`.
- Changed: `pinilla_ramos_et_al_2024.duration_model` uses all of the values of array inputs. It
  previously used only the first value.
- Fixed: `AbrahamsonSilva1996.interp` and `calc_ln_dur_incr` no longer modify the `nias` array that
  is passed in, and accept integer values.
- Added: `ims` argument for all ground motion models (`GroundMotionModel` subclasses) to compute only
  the requested intensity measures: "pga", "pgv", individual spectral periods such as "psa_1p000" (1.0 s),
  "psa_ngawest2_21" (the 21 NGA-West2 comparison periods), and "psa_all" (all periods).
  `periods` and `spec_accels` contain only the computed periods, and `psa_ims` gives their names.
- Fixed: Boore et al. (2014) used the global relation for the reference depth to 1.0 km/s for the
  Japan region. It now uses the Japan relation, which changes the basin term at periods of 0.65 s
  and longer for `region="japan"` with `depth_1_0` specified.
- Changed: `ChiouYoungs2014.calc_depth_1_0` accepts an array of regions.
- Fixed: the warning for a scalar value above a parameter's recommended limit showed
  "{self.max}" instead of the limit.
- Fixed: Pezeshk et al. (2011) used the standard deviation in log10 units as a natural log standard
  deviation. It is now converted to natural log units (multiplied by ln(10)), as in OpenQuake,
  e.g., 0.58 instead of 0.25 for PGA at M6.
- Fixed: Atkinson and Boore (2006) used the hard-rock coefficients only for `v_s30` of zero, so
  hard-rock sites used the B/C coefficients with the site amplification extrapolated to their
  velocity. Sites with `v_s30` of 2000 m/s or greater now use the hard-rock coefficients without site
  amplification, as in the paper and OpenQuake. At 2000 m/s, PGA increases by a factor of about 1.4
  to 2.1 and PSA at 1 s by about 1.5.
- Fixed: Atkinson and Boore (2006) divided PGV and PGD by g along with PGA and PSA, so `pgv` and
  `pgd` were 980.665 times too small. They are now in cm/sec and cm, as for the other models (e.g.,
  PGV of 44 cm/sec for M7 at 10 km on B/C).
- Fixed: Atkinson and Boore (2006) used the standard deviation of 0.30, which is in log10 units, as
  a natural log standard deviation. It is now 0.30 ln(10) = 0.691, as in nshmp-haz and OpenQuake.
- Fixed: Atkinson and Boore (2006) left out the "+ b_2" in the nonlinear site coefficient for
  180 < Vs30 <= 300 m/s (Eq. 8), so the site term was discontinuous at 180 and 300 m/s. This changes
  the results for Vs30 in this range, e.g., PGA decreases by about 28% for M7 at 10 km with
  Vs30 = 250 m/s.
- Fixed: `PinillaRamosEtAl2024.d5x_median` scaled the D5-75 median by approximate factors (e.g.,
  0.25 for D5-10 and 1.80 for D5-95), and `d5x_sigma` returned the D5-75 standard deviation for all
  thresholds. Both now use the conditional model of `duration_for_energy`. For example, D5-10 is
  about 4% of D5-75 instead of 25%, and D5-95 is about 2.0 to 2.6 times D5-75 instead of 1.8.
  Thresholds from "D5-10" to "D5-95" in steps of 5% are supported.
- Fixed: Pinilla-Ramos et al. (2023) had a0 = 1 instead of 0 for the 0.75 row of the conditional
  model, which doubled the durations from `duration_for_energy` for energies between about 0.725
  and 0.775 other than 0.75. These energies now give the D5-75 durations.
- Fixed: `AbrahamsonSilva1996.interp` added the increment for the 5 to 75% duration twice, which
  made the interpolated durations about 0.9% too large. `interp(0.75)` now equals `duration`, and
  the results match the reference spreadsheet to within its rounding.
- Fixed: the `adjust_c4` argument of Abrahamson, Gregor, and Addo (2016) was stored but not used. It
  is now added to C_4 (10 km) in the finite-fault distance term. The default of 0 gives the same
  results as before.
- Fixed: Campbell and Bozorgnia (2014) did not set PSA at periods shorter than 0.25 s to PGA when
  it is less than PGA, as specified by the model. This increases the short-period PSA for sites with
  strong nonlinear site response, e.g., by up to about 40% at 0.05 to 0.075 s for Vs30 = 150 m/s.
  The standard deviations are unchanged.
- Fixed: the Hermkes, Kuehn, and Riggelsen (2014) model data in the repository was a Dropbox web page
  instead of the model data, because the download URL used `dl=0`. The data (164 MB, Git LFS) is
  restored, the download uses `dl=1` and checks that the downloaded file is the model data, and a
  Git LFS pointer file (a clone without Git LFS) is replaced by the download. The data is not
  included in the package and is downloaded when the model is first imported.
- Fixed: Campbell and Bozorgnia (2014) used the global anelastic attenuation coefficient
  (Δc20 = 0) for `region="china"` because the region was compared with a list. It now uses the
  China coefficient (Δc20,CH), which increases the response at distances greater than 80 km
  (e.g., PGA by about 30% at 150 km for M6.5 and Vs30 = 760 m/s).
- Fixed: Afshari and Stewart (2016) applied the maximum basin term (a depth differential of 200 m)
  for `depth_1_0=np.nan`. An unknown depth (NaN) now has no basin term, as for `depth_1_0=None`, which
  allows arrays of scenarios with some unknown depths.
- Fixed: Afshari and Stewart (2016) passed the mechanism as the basin region, so the California
  relation for the mean depth to 1.0 km/s was always used. It now has an optional `region`
  parameter ("california", "global", or "japan"; default "california") and uses the Japan
  relation for `region="japan"`.
- Fixed: Afshari and Stewart (2016) computed the basin depth differential in km instead of m,
  which made the basin term about 1,000 times too small. Figure 12 of the paper shows that c_5
  applies to the differential in m (Table 2 lists c_5 in 1/km), consistent with the 200 m limit.
- Fixed: Afshari and Stewart (2016) used b_1 = 6.188 instead of 6.579 for the D20-80 duration of
  reverse earthquakes (Table 1 of the paper).

# 0.8.0 (2025-07-24)
- Added: Pinilla-Ramos et al. (2023) model for duration of crustal earthquakes
- Added: Pinilla-Ramos et al. (2024) model for duration of subduction earthquakes
- Added: Stafford (2017) model for FAS correlation

# 0.7.3 (2025-03-12)
-   Fixed Bayless and Abrahamson (2018) correlation model.

# 0.7.1 (2025-03-05)

-   Added compatibility with numpy 2.0

# 0.7.0 (2024-04-24)

-   Added: Abrahamson and Bhasin (2020)
-   Changed to Hatch build system

# 0.6.6 (2023-12-11)

-   Added: Return tau and phi in the standard deviation calculations

# 0.6.5 (2022-09-16)

-   Added: Afshari and Stewart (2016) duration model
-   Added: Kempton and Stewart (2006) duration model

# 0.6.4 (2022-01-24)

-   Added: Bayless and Abrahamson (2019)

# 0.6.3 (2021-12-08)

-   Fixed: error in ASK14 on a7 term

# 0.6.2 (2021-10-19)

-   Changed: Move site amplification to static functions on some GMPEs

# 0.6.1 (2020-06-03)

-   Added Coppersmith & Bommer (2014) model for Hanford
-   Factored tests

# 0.6.0 (2019-08-12)

-   Added Abrahamson, Gregor, Addo (2014)
-   Added Abrahamson & Gulerce (2011)
-   Added conditional mean spectra models.
-   Added Scenario objects.
-   Added typing for all classes.

# 0.4.0 (2016-04-08)

-   Added Hermkes et al. (2014).
-   Improved documentation.
-   Added Baker & Jayaram (2008), Kishida (2017)

# 0.3.2 (2016-03-30) {#section-1}

-   Nothing changed yet.

# 0.3.1 (2016-03-30) {#section-2}

-   First release on PyPI.
