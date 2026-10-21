# MMM transformation survey (reference survey)

This is a **transformation-oriented comparison**, not a benchmark of estimation quality. Mathematical ideas are reimplemented independently; no proprietary implementation is copied. Parameters, normalization, order and estimation assumptions differ across frameworks.

| Framework | Documented transformation families | HypoForge status |
|---|---|---|
| **Meta Robyn** | Geometric and Weibull adstock; Hill saturation; trend/season/holiday controls | Geometric and a discrete Weibull-kernel variant; Hill already in core |
| **Google Meridian** | Geometric or binomial adstock; Hill saturation; configurable adstock/Hill ordering; reach × saturated frequency | Geometric and reach-frequency building block; binomial and ordering adapter pending |
| **Google LightweightMMM** | Adstock + exponent, carryover + exponent, Hill-adstock; Fourier seasonality | Geometric, power saturation, annual Fourier; carryover peak shape is not claimed equivalent |
| **PyMC-Marketing** | Geometric, delayed, binomial, Weibull adstock; logistic, inverse logistic, Hill, Michaelis–Menten, root, tanh saturation | Geometric, delayed-kernel variant, Weibull, logistic, Michaelis–Menten, power/root, tanh; others pending |

## Implemented standalone functions

- `geometric_adstock(x, alpha, l_max, normalize=True)`: finite causal geometric lag convolution, with available-history normalization.
- `delayed_adstock(x, alpha, peak, l_max, normalize=True)`: discrete peaked lag weights `alpha**abs(lag-peak)`; **not an exact reproduction** of every framework's delayed-adstock parameterization.
- `weibull_adstock(x, shape, scale, l_max, normalize=True)`: discrete Weibull survival differences, not an exact Robyn Weibull CDF/PDF compatibility mode.
- `logistic_saturation(x, lam)`: `tanh(lam*x/2)`, zero at zero, bounded by one.
- `michaelis_menten(x, half_saturation, maximum)`: `maximum*x/(half_saturation+x)`.
- `power_saturation(x, exponent)`: `x**exponent`, exponent in (0,1].
- `tanh_saturation(x, scale)`: `tanh(x/scale)`.
- `reach_frequency(reach, frequency, half_saturation, slope)`: `reach * Hill(frequency)`, without adstock.

These operations are *hypothesis transformations*, not claims that one of the source packages uses identical kernels, scaling, priors or defaults.

## Next candidates, not yet implemented

Binomial adstock; Robyn-compatible Weibull PDF/CDF options; inverse-scaled logistic; baselined tanh; composite transform specifications with explicit order; fold-fitted hyperparameter selection; parameter bounds and monotonicity diagnostics; support for panel/group-aware convolution; calibration and uncertainty estimation. **Do not add every transform blindly:** candidate-space size and false discovery risk grow rapidly.

## Primary references

1. Meta Robyn features: https://facebookexperimental.github.io/Robyn/docs/features/
2. Google Meridian media saturation/lagging: https://developers.google.com/meridian/docs/advanced-modeling/media-saturation-lagging
3. Google Meridian reach and frequency: https://developers.google.com/meridian/docs/advanced-modeling/reach-frequency
4. Google LightweightMMM API: https://lightweight-mmm.readthedocs.io/en/latest/api.html
5. Google LightweightMMM source: https://github.com/google/lightweight_mmm/blob/main/lightweight_mmm/models.py
6. PyMC-Marketing transformation API: https://www.pymc-marketing.io/en/latest/api/generated/pymc_marketing.mmm.transformers.html
7. PyMC-Marketing model overview: https://github.com/pymc-labs/pymc-marketing

Source links describe external projects; they do not imply endorsement or API equivalence.
