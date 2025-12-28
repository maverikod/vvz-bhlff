<!-- markdownlint-disable MD009 -->
<!--
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Assessment of the amorphous-state topic in the 7D phase-field theory corpus.

This note summarizes (i) how the current corpus treats crystal ↔ amorphous
within a single, uniform “passport / regime” apparatus, and (ii) what
practice-validated theories in condensed matter and rheology look most similar
in approach (dimensionless regime criteria + inverse mapping from experiments),
as well as where the current corpus is still weaker than state-of-the-art.

Theoretical Background:
    In the corpus, the amorphous regime is defined dynamically (spectral
    coherence breakdown) and is diagnosed via a dimensionless control parameter
    built from experimental loss data (Q^{-1}) and characteristic frequencies.

Example:
    Use the corpus to compute a regime label from DMA/mechanical spectroscopy:
        1) Measure Q^{-1}(ω, T)
        2) Compute Xi_hat = (ω/ω0) * Q^{-1}
        3) Classify the regime by Xi_hat thresholds
-->

## Context: what “amorphous” means in the current corpus

In the current corpus (notably the `C.11` block embedded inside `7d-112`),
the amorphous state is *not* defined by geometric language (“no lattice”),
but by a **dynamic/spectral regime**:

- **Crystal regime**: discrete spectrum, long-lived modes, stationary connectivity
- **Amorphous regime**: quasi-continuous spectrum, short-lived modes, dynamically
  reconfiguring connectivity

The key point is uniformity: **crystal and amorphous are expressed in the same
variables**, and differ by **value ranges and spectral structure**, not by using
two unrelated models.


## Unified apparatus (crystal ↔ amorphous) is the main scientific step

### One regime axis, one passport language

The corpus uses a single dimensionless control parameter (a regime axis):

- **Xi_hat**: \(\widehat{\Xi}_E = (\omega/\omega_0)\,Q^{-1}\)

where:

- \(\omega\) is the external excitation frequency,
- \(Q^{-1}\) is the measured loss (e.g., \(\tan\delta\) in DMA-like settings),
- \(\omega_0\) is a characteristic internal frequency scale for the material.

Then:

- **Crystal** corresponds to \(\widehat{\Xi}_E \ll 1\)
- **Amorphous** corresponds to \(\widehat{\Xi}_E \lesssim 1\)
- Larger values are interpreted as entering increasingly dissipative/connectedness-
  suppressed regimes (depending on the corpus’ classification table).

This is scientifically valuable because it converts “phase talk” into a **testable,
algorithmic classification** driven by the same computed number for both regimes.


### Same bridge to experiment (inverse problem)

The corpus insists on an **inverse mapping from measured data**:

- Start from measurements: \(Q^{-1}(\omega, T)\)
- Compute: \(\widehat{\Xi}_E(\omega, T) = (\omega/\omega_0)\,Q^{-1}(\omega, T)\)
- Use thresholds to assign a regime label and interpret transitions.

This makes the discussion *operational*: it is no longer “only theory” because
it prescribes what to compute from standard experimental observables.


## Practice-confirmed theories with a similar *approach*

There is no universally accepted single theory that derives *everything* about
glass/amorphization across all materials without calibration. However, there are
highly practice-validated frameworks that are similar in *approach* (dimensionless
regime criteria + inverse mapping from experiments).

### Closest in spirit (engineering-level, robust in practice)

- **Deborah number / ωτ criteria**
  - Widely used in rheology/viscoelasticity: regime changes occur near
    \( \omega \tau \sim 1 \).
  - This is very close in *logic* to using \(\widehat{\Xi}_E\) as a regime axis:
    “the system cannot relax within the forcing period”.

- **DMA + time–temperature superposition (TTS)**
  - Uses measured \(G'(\omega,T)\), \(G''(\omega,T)\), \(\tan\delta(\omega,T)\)
    to build master curves and extract relaxation times/scales.
  - Highly validated in polymers, glasses, composites.
  - Extremely “passport-like”: the material is characterized by measured curves
    and derived dimensionless scaling variables.

- **Generalized viscoelastic models (Maxwell/Kelvin-Voigt/Prony, fractional rheology)**
  - Provide unified, testable response descriptions across regimes, but are often
    phenomenological (parameters fitted to data).

### Higher-level glass theories (strong physics, less universal)

- **Mode-Coupling Theory (MCT)**
  - Useful for parts of supercooled-liquid dynamics; struggles near real \(T_g\)
    without extensions and material-specific tuning.

- **RFOT / energy landscape / trap models**
  - Provide deep conceptual structure and sometimes quantitative fits, but do not
    offer a universal passport mapping for arbitrary materials without calibration.

- **STZ / SGR and related plasticity models**
  - Strong for amorphous plasticity/yielding; still parameterized and domain-specific.

- **Jamming framework**
  - Very successful for athermal/granular systems; different control parameters
    and not a universal glass theory for thermal materials.


## Scientific value: what is genuinely strong already

- **Uniform formalism across states**:
  - One regime axis + one passport format reduces fragmentation and makes the
    theory easier to test and falsify.

- **Operational criterion**:
  - A clear algorithm exists: compute \(\widehat{\Xi}_E\) from measurable quantities.

- **Regime-by-regime validity (and explicit limits of “classical fits”)**:
  - The corpus frames “bad fit” as a regime boundary, which is a scientifically
    productive viewpoint when handled carefully.


## Main weaknesses (what prevents “higher-than-classical” predictive power today)

### 1) The ω0 calibration risk (possible circularity)

If \(\omega_0\) is fixed using quantities that already encode glass-transition
conventions (e.g., canonical \(\eta_g\) definitions), then the method can become
partly circular: “glass-ness” sneaks in via the chosen calibration.

To strengthen the scientific claim, the corpus needs a clearly stated hierarchy:

- How \(\omega_0\) is obtained **independently** (or with minimal, explicit,
  cross-material assumptions),
- Sensitivity analysis: how stable regime boundaries are under plausible variations
  of \(\omega_0\).

### 2) Universality of numeric thresholds must be demonstrated

Ranges like \(Q^{-1}\sim 10^{-2}\ldots 10^{-1}\) and \(\widehat{\Xi}_E\sim 10^{-2}\ldots 1\)
may be typical, but they must be validated across:

- oxide glasses, amorphous metals, polymers, etc.,
- different frequency windows and temperatures.

Otherwise, the passport risks becoming a “typicality statement” instead of a universal
criterion.

### 3) Regime classification ≠ microphysical prediction

Today, the strongest part is **classification and inversion from data**.
What is still missing for a chemistry-grade predictive theory:

- Predict \(T_g\), relaxation spectra, and transitions from structure/composition
  with minimal or no fitting,
- A validated mapping from network/ensemble descriptors (connectivity/spectrum)
  to classical observables (\(T, \rho, p\)) that is stable across material classes.


## Bottom line

The “amorphous state” topic has moved beyond pure theorizing because it is now
structured as:

- a **unified regime apparatus** shared with the crystal description,
- an **algorithmic passport** using measurable inputs,
- an explicit **inverse problem** (experiment → regime variables).

The closest real-world analogs in *approach* are rheology/DMA frameworks
(Deborah number, ωτ criteria, TTS), which are strongly practice-validated.
The corpus becomes genuinely “higher-level” once \(\omega_0\) is anchored
non-circularly and the passport thresholds are demonstrated to be transferable
across multiple amorphous material classes.


