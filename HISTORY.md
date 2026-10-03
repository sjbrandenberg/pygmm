---
title: History
---

# Unreleased
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
