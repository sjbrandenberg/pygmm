---
title: History
---

# Unreleased
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
- Fixed: Pinilla-Ramos et al. (2023) had a0 = 1 instead of 0 for the 0.75 row of the conditional
  model, which doubled the durations from `duration_for_energy` for energies between about 0.725
  and 0.775 other than 0.75. These energies now give the D5-75 durations.
- Fixed: `AbrahamsonSilva1996.interp` added the increment for the 5 to 75% duration twice, which
  made the interpolated durations about 0.9% too large. `interp(0.75)` now equals `duration`, and
  the results match the reference spreadsheet to within its rounding.
- Fixed: Campbell and Bozorgnia (2014) used the global anelastic attenuation coefficient
  (Δc20 = 0) for `region="china"` because the region was compared with a list. It now uses the
  China coefficient (Δc20,CH), which increases the response at distances greater than 80 km
  (e.g., PGA by about 30% at 150 km for M6.5 and Vs30 = 760 m/s).
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
