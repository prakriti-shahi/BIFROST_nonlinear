import numpy as np
from scipy import constants as const

# ── Raman model constants for fused silica ────────────────────────────────
RAMAN_TAU1_SI   = 12.2e-15   # s — phonon period
RAMAN_TAU2_SI   = 32.0e-15   # s — coherence decay time

RAMAN_FR_SI     = 0.18       # dimensionless — fractional Raman contribution
RAMAN_OMEGA_R   = 1.0 / RAMAN_TAU1_SI  # rad/s, peak Raman shift ≈ 13.2 THz

def h_R_time(t: np.ndarray) -> np.ndarray:
    """
    Raman impulse response function hR(t) for fused silica
    (Blow & Wood 1989 two-phonon model).

    Correctly zero for t < 0 (causal).
    Normalised so that integral over [0,inf) = 1.

    Parameters
    ----------
    t : np.ndarray  Time array (s). May include negative values.

    Returns
    -------
    np.ndarray  hR(t) values, zero for t <= 0.
    """
    tau1 = RAMAN_TAU1_SI
    tau2 = RAMAN_TAU2_SI
    prefactor = (tau1**2 + tau2**2) / (tau1 * tau2**2)
    h = prefactor * np.exp(-t / tau2) * np.sin(t / tau1)
    h[t <= 0] = 0.0  # enforce causality
    return h

def h_R_freq(Omega):
    """
    Analytical Fourier transform of h_R(t), defined with the +iΩt convention:

        h̃_R(Ω) = ∫₀^∞ h_R(t) e^{+iΩt} dt
                = [(τ₁²+τ₂²)/(τ₁²τ₂²)] / [(1/τ₂ − iΩ)² + 1/τ₁²]

    Im[h̃_R(Ω)] > 0 for Ω > 0  (Stokes gain).
    Im[h̃_R(Ω)] < 0 for Ω < 0  (anti-Stokes loss).

    Parameters
    ----------
    Omega : array_like
        Angular frequency shift Ω = ω_pump − ω_signal (rad/s).

    Returns
    -------
    np.ndarray  (complex, dimensionless)

    References
    ----------
    Agrawal, Nonlinear Fiber Optics, 6th ed. (2019), Eq. (2.3.43).
    Blow & Wood, IEEE J. Quantum Electron. 25, 2665 (1989), Eq. (2).
    """
    Omega = np.asarray(Omega, dtype=complex)
    tau1 = RAMAN_TAU1_SI   # 12.2e-15 s
    tau2 = RAMAN_TAU2_SI   # 32.0e-15 s

    numerator   = (tau1**2 + tau2**2) / (tau1**2 * tau2**2)   # s⁻²
    denominator = (1/tau2 - 1j*Omega)**2 + (1/tau1)**2        # s⁻²

    return numerator / denominator   

def g_R(Omega: np.ndarray, gamma: float,
        fR: float = RAMAN_FR_SI) -> np.ndarray:
    """
    Raman gain coefficient gR(Ω) for fused silica (or doped silica).

    gR(Ω) = 2 × γ × fR × Im[h̃R(Ω)]

    Parameters
    ----------
    Omega  : np.ndarray  Raman shift in rad/s. Positive = Stokes.
    gamma  : float       Nonlinear coefficient of the fiber (W⁻¹m⁻¹).
    fR     : float       Raman fraction. Default: 0.18 (fused silica).

    Returns

    ------
    np.ndarray  gR(Ω) in m/W.
    """
    return 2.0 * gamma * fR * np.imag(h_R_freq(Omega))

def g_R_from_wavelengths(lambda_pump: float, lambda_signal: float,
                         gamma: float) -> float:
    """
    Raman gain at a signal wavelength, given a pump wavelength.

    Parameters
    ----------
    lambda_pump   : float  Pump wavelength (m).
    lambda_signal : float  Signal (Stokes) wavelength (m).
    gamma         : float  Fiber nonlinear coefficient (W⁻¹m⁻¹).

    Returns
    -------
    float  gR in m/W.
    """
    c = 299792458.0  # m/s
    omega_pump   = 2 * np.pi * c / lambda_pump
    omega_signal = 2 * np.pi * c / lambda_signal
    Omega = omega_pump - omega_signal  # positive for Stokes
    return float(g_R(np.array([Omega]), gamma)[0])

def thermal_photon_number(Omega: np.ndarray, T: float) -> np.ndarray:
    """
    Bose-Einstein thermal phonon occupancy nth(Ω,T).

    Parameters
    ----------
    Omega : np.ndarray  Raman shift in rad/s (positive).
    T     : float       Temperature in Kelvin.

    Returns
    -------
    np.ndarray  nth, dimensionless.
    """
    hbar = const.hbar   # 1.0546e-34 J·s
    kB   = const.k      # 1.3806e-23 J/K
    x = hbar * np.abs(Omega) / (kB * T)
    # Protect against overflow/underflow
    return np.where(x > 700, 0.0, 1.0 / (np.expm1(x)))

def spRam_photon_rate_density(Omega: np.ndarray,
                              omega_pump: float,
                              P_pump: float,
                              L: float,
                              gamma: float,
                              T: float = 300.0,
                              sideband: str = 'stokes') -> np.ndarray:
    """
    Spontaneous Raman noise photon spectral density (photons/s per rad/s bandwidth)
    at the output of a fiber with a CW undepleted pump.

    Parameters
    ----------
    Omega      : np.ndarray  Raman shift Ω = ωp − ωs (rad/s), positive values.
    omega_pump : float       Pump angular frequency ωp (rad/s).
    P_pump     : float       Pump power (W).
    L          : float       Fiber length (m).
    gamma      : float       Fiber nonlinear coefficient γ (W⁻¹m⁻¹).
    T          : float       Temperature (K). Default 300 K.
    sideband   : str         'stokes' or 'antistokes'.

    Returns
    -------
    np.ndarray  Photon rate density dṄ/dΩ [photons/(s·rad/s)] at Stokes
                or anti-Stokes frequencies.
    """
    hbar = const.hbar
    gR = g_R(Omega, gamma)          # m/W, shape (len(Omega),)
    nth = thermal_photon_number(Omega, T)  # dimensionless

    if sideband.lower() == 'stokes':
        omega_sig = omega_pump - Omega  # Stokes: red-shifted
        photon_energy = hbar * omega_sig
        bose_factor = nth + 1.0
    elif sideband.lower() == 'antistokes':
        omega_sig = omega_pump + Omega  # anti-Stokes: blue-shifted
        photon_energy = hbar * omega_sig
        bose_factor = nth
    else:
        raise ValueError("sideband must be 'stokes' or 'antistokes'.")

    # dṄ/dΩ = P_pump × L × gR(Ω) / (ℏωs) × bose_factor
    rate_density = P_pump * L * gR * bose_factor / (2 * np.pi)
    return rate_density

def spRam_noise_in_channel(lambda_pump: float,
                           lambda_channel: float,
                           delta_lambda: float,
                           P_pump: float,
                           L: float,
                           gamma: float,
                           T: float = 300.0,
                           n_points: int = 200) -> dict:
    """
    Spontaneous Raman noise photon rate (photons/s) in a spectral channel.
    Integrates over the channel bandwidth using Simpson's rule.

    Parameters
    ----------
    lambda_pump    : float  Pump (reference) wavelength (m).
    lambda_channel : float  Centre wavelength of the quantum channel (m).
    delta_lambda   : float  FWHM bandwidth of the quantum channel (m).
    P_pump         : float  Pump power (W).
    L              : float  Fiber length (m).
    gamma          : float  Fiber nonlinear coefficient γ (W⁻¹m⁻¹).
    T              : float  Temperature (K). Default 300 K.
    n_points       : int    Integration grid points. Default 200.

    Returns
    -------
    dict with keys:
       'stokes_photons_per_sec'    : float
       'antistokes_photons_per_sec': float
       'stokes_power_W'            : float
       'antistokes_power_W'        : float
       'dominant_sideband'         : str
       'Omega_centre_THz'          : float
    """
    from scipy.integrate import simpson
    c = 299792458.0
    hbar = const.hbar

    omega_pump = 2 * np.pi * c / lambda_pump
    # Build wavelength grid covering the channel bandwidth
    lam_lo = lambda_channel - delta_lambda / 2
    lam_hi = lambda_channel + delta_lambda / 2
    lam_grid = np.linspace(lam_lo, lam_hi, n_points)
    omega_grid = 2 * np.pi * c / lam_grid
    Omega_grid = np.abs(omega_pump - omega_grid)

    # Determine if channel is on Stokes or anti-Stokes side
    if lambda_channel > lambda_pump:
        sideband = 'stokes'
    else:
        sideband = 'antistokes'

    rate_dens = spRam_photon_rate_density(
        Omega_grid, omega_pump, P_pump, L, gamma, T, sideband
    )

    # dω = (2πc/λ²) dλ — Jacobian for converting from dλ to dΩ
    domega_dlam = 2 * np.pi * c / lam_grid**2
    photons_per_sec = float(simpson(rate_dens * domega_dlam, lam_grid))

    # Also compute noise power (W) = photons/s × ℏωs
    omega_ch = 2 * np.pi * c / lambda_channel
    noise_power_W = photons_per_sec * hbar * omega_ch

    Omega_centre = np.abs(omega_pump - omega_ch)

    result = {
        'stokes_photons_per_sec'     if sideband == 'stokes' else
        'antistokes_photons_per_sec' : photons_per_sec,
        'stokes_power_W'             if sideband == 'stokes' else
        'antistokes_power_W'         : noise_power_W,
        'dominant_sideband'          : sideband,
        'Omega_centre_THz'           : Omega_centre / (2*np.pi*1e12)
    }
    # Fill the missing sideband with zeros for completeness
    for k in ['stokes_photons_per_sec','antistokes_photons_per_sec',
              'stokes_power_W','antistokes_power_W']:
        result.setdefault(k, 0.0)
    return result

def depletion_gain_param(g_R_val: float, P_pump: float, L: float) -> float:
    """
    Dimensionless gain parameter x = g_R * P_pump * L for the pump-depletion
    correction.

    This is the argument of the exponential in the corrected noise formula.
    When x ≪ 1 the undepleted approximation holds; when x ≳ 0.1 the
    correction is needed.

    Rule of thumb:
        x < 0.02   →  correction < 1%    (safe to ignore)
        x > 0.19   →  correction > 10%   (should use pump_depletion=True)
        x > 1.0    →  correction > 70%   (undepleted is significantly wrong)

    Parameters
    ----------
    g_R_val : float
        Raman gain coefficient at the relevant frequency (W⁻¹ m⁻¹).
        Use the peak value g_R(Ω_R) for a worst-case estimate.
    P_pump : float
        Pump power (W).
    L : float
        Fiber length (m).

    Returns
    -------
    float
        Dimensionless gain parameter x = g_R * P_pump * L.
    """
    return float(g_R_val) * float(P_pump) * float(L)

def check_depletion_validity(g_R_peak: float, P_pump: float, L: float,
                              threshold: float = 0.10) -> dict:
    """
    Check whether the undepleted-pump approximation is valid and print a
    diagnostic.

    Parameters
    ----------
    g_R_peak : float
        Peak Raman gain coefficient (W⁻¹ m⁻¹).  Use ``raman.g_R`` at
        Ω_R = 13.2 THz × 2π, or equivalently ``raman_tabulated.g_R_tabulated``
        at the same frequency.
    P_pump : float
        Pump power (W).
    L : float
        Fiber length (m).
    threshold : float, optional
        Fractional correction above which a warning is issued (default 0.10
        = 10%).

    Returns
    -------
    dict with keys:
        'x'               : float  — gain parameter g_R * P * L
        'correction_ratio': float  — [exp(x)−1]/x  (≥ 1)
        'correction_pct'  : float  — percentage overcounting of undepleted model
        'valid'           : bool   — True if correction < threshold
        'recommendation'  : str

    Example
    -------
    >>> import raman
    >>> result = check_depletion_validity(
    ...     g_R_peak = raman.g_R(np.array([13.2e12*2*np.pi]), gamma=1.3e-3)[0],
    ...     P_pump   = 1e-2,   # 10 mW
    ...     L        = 25e3)   # 25 km
    >>> print(result['recommendation'])
    """
    import warnings as _warnings
    x = depletion_gain_param(g_R_peak, P_pump, L)
    ratio = float(np.expm1(x) / x) if x > 1e-10 else 1.0
    pct   = (ratio - 1.0) * 100.0
    valid = pct < threshold * 100.0

    rec = ("Undepleted approximation valid."
           if valid else
           f"Undepleted overcounts by {pct:.1f}%; use pump_depletion=True.")

    result = dict(x=x, correction_ratio=ratio, correction_pct=pct,
                  valid=valid, recommendation=rec)

    if not valid:
        _warnings.warn(
            f"Pump depletion correction needed: g_R*P_p*L = {x:.3f}, "
            f"undepleted overcounts noise by {pct:.1f}%. "
            f"Pass pump_depletion=True to spRam_noise_in_channel / calcSpRamNoise.",
            UserWarning, stacklevel=2)

    return result

def _spRam_rate_density_core(gR: np.ndarray,
                              P_pump: float,
                              L: float,
                              factor: np.ndarray,
                              pump_depletion: bool = False) -> np.ndarray:
    """
    Core rate density calculation, with or without pump-depletion correction.

    Parameters
    ----------
    gR : np.ndarray
        Raman gain at each Ω (W⁻¹ m⁻¹).  Must be ≥ 0.
    P_pump : float
        Pump power (W).
    L : float
        Fiber length (m).
    factor : np.ndarray
        Thermal occupancy factor: (n_th+1) for Stokes, n_th for anti-Stokes.
    pump_depletion : bool
        If False (default), use the linear undepleted formula.
        If True, use the exponential corrected formula.

    Returns
    -------
    np.ndarray
        dṄ/dΩ in photons s⁻¹ (rad/s)⁻¹, before the floor at 0.
    """
    if not pump_depletion:
        # Undepleted: rate ∝ g_R * P_pump * L
        return P_pump * L * gR * factor / (2 * np.pi)
    else:
        # Corrected: replace g_R * P_pump * L  →  expm1(g_R * P_pump * L)
        #
        # Rate = (n_th+1) * [exp(g_R*P_p*L) - 1] * Dnu / (2π)
        # where the Dnu integral is handled by the caller.
        #
        # For g_R ≈ 0 (outside Raman band), expm1(0)/g_R is 0/0 → use
        # L'Hopital limit = P_pump * L, so both branches agree at g_R = 0.
        x = gR * P_pump * L   # dimensionless gain parameter, element-wise
        # Numerically stable via expm1; handles both small and large x
        corrected_PL = np.where(
            gR > 1e-25,
            np.expm1(x) / gR,   # [exp(x)-1]/g_R  = effective length × pump
            P_pump * L          # L'Hopital limit at g_R → 0
        )
        return corrected_PL * factor / (2 * np.pi)
    


print("Finished running")

