"""
spinning.py — Analytical Jones matrices for spun optical fibers.

Implements the spinning formalism from Sections 12.1–12.2 of the
Polarization Compensation Project Notes and the Floquet solution
for periodic spin rates (Britton, 3/13/2026).

Two regimes are supported:

1. Constant Spin Rate, Constant Birefringence (CSRCB)
   Closed-form Jones matrix via the matrix/Jones calculus methods
   (Eqns. 76/91 in the project notes; Eqn. 6 in Huang et al. 2024).

2. Sinusoidal Spin Rate, Constant Birefringence (Floquet)
   Analytical approximate Jones matrix valid when the modulation index
   m = 2*xi0/omega >> 1 and omega >= beta.  Uses the Jacobi–Anger /
   Bessel-function averaging result (Eqn. 21 in the Floquet notes).

Both functions return 2×2 complex Jones matrices in the x-y (linear) basis.  
"""

import numpy as np
from scipy.special import jv  # Bessel function of the first kind

pi = np.pi


# =========================================================================
#  Rotation matrix helper
# =========================================================================
def _rotation_matrix(angle):
    """2×2 Jones rotation matrix R(angle)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s], [-s, c]])


# =========================================================================
#  1.  CSRCB — Constant Spin Rate, Constant Birefringence
# =========================================================================
def calc_J_CSRCB(delta, xi0, L, alpha_circ=0.0):
    r"""
    Jones matrix for a spun fiber with constant spin rate and constant
    birefringence.

    Derived via two independent methods (matrix exponentiation and
    Jones differential calculus) in Sec. 12.1 of the project notes.
    Both yield the same result, Eqn. (76)/(91):


    Parameters
    ----------
    delta : float
        Intrinsic linear birefringence :math:`\delta_0 = \beta_y - \beta_x`
        (rad/m).  This is the total linear birefringence from all sources
        (CNC + ATS + BND) — not including twist, which is a circular
        birefringence and enters through ``alpha_circ``.
    xi0 : float
        Constant spin rate (rad/m).  Positive = counter-clockwise when
        looking along +z.
    L : float
        Length of the fiber segment (m).
    alpha_circ : float, optional
        Circular birefringence parameter.  Default 0.

    Returns
    -------
    J : np.ndarray, shape (2, 2), complex
        Jones matrix in the x-y linear basis.

    Notes
    -----
    * When xi0 = 0 and alpha_circ = 0 the result reduces to the
      standard diagonal waveplate Jones matrix.
    * The spin enters the optics as a rotation of the birefringence axes, 
      not as a twist-induced circular birefringence.  Physical twist 
      birefringence (photoelastic shear) should be passed via alpha_circ.
    """
    # Generalized Rabi-like frequency  (Eqn. 88 notation: Omega/2)
    Omega_half = np.sqrt((delta / 2) ** 2 + ((alpha_circ + 2 * xi0) / 2) ** 2)

    if Omega_half == 0:
        # No birefringence, no spin → identity
        return np.eye(2, dtype=complex)

    cosG = np.cos(Omega_half * L)
    sinG = np.sin(Omega_half * L)

    detuning = (alpha_circ + 2 * xi0) / (2 * Omega_half)
    J_rot = np.array([
        [cosG - 1j * (delta / (2 * Omega_half)) * sinG,
         -detuning * sinG],
        [detuning * sinG,
         cosG + 1j * (delta / (2 * Omega_half)) * sinG]
    ], dtype=complex)

    R = _rotation_matrix(xi0 * L)
    return R @ J_rot


# =========================================================================
#  2.  Sinusoidal spin — Floquet / Bessel averaging
# =========================================================================
def calc_J_sinusoidal(delta, xi0, omega, L, alpha_circ=0.0,
                      n_bessel_terms=0):
    r"""
    Approximate Jones matrix for a fiber with sinusoidal spin rate and constant
    linear birefringence (and optionally constant circular birefringence).
    Valid in the Floquet regime.

    The result is (Eqn. 21 of the Floquet notes).

    Parameters
    ----------
    delta : float
        Intrinsic linear birefringence (rad/m).
    xi0 : float
        Spin-rate amplitude (rad/m).
    omega : float
        Spatial angular frequency of the spin modulation (rad/m).
        The spatial period is :math:`T = 2\pi/\omega`.
    L : float
        Length of the fiber segment (m).
    alpha_circ : float, optional
        Circular birefringence (rad/m).  Default 0.
    n_bessel_terms : int, optional
        Number of additional Bessel harmonics (beyond n=0) to include
        in the secular Hamiltonian.  Default 0 keeps only the leading
        J_0(m) term.  Setting this to a positive integer includes
        corrections from the harmonics via second-order perturbation 
        theory (Floquet–Magnus).
        Not yet implemented — reserved for future use.

    Returns
    -------
    J : np.ndarray, shape (2, 2), complex
        Jones matrix in the x-y linear basis.
    info : dict
        Dictionary with diagnostic quantities:

    Raises
    ------
    ValueError
        If omega <= 0.
    """
    if omega <= 0:
        raise ValueError("omega must be positive; got {:.4e}".format(omega))

    if n_bessel_terms != 0:
        raise NotImplementedError(
            "Higher-order Bessel corrections not yet implemented. "
            "Use n_bessel_terms=0."
        )

    m = 2 * xi0 / omega          # modulation index
    J0m = float(jv(0, m))        # J_0(m)
    beta_eff = delta * J0m       # effective birefringence

    a = beta_eff * L / 2   # half the accumulated effective phase

    # Secular evolution in the circular basis (Eqn. 20)
    U_secular = np.array([
        [np.cos(a), -1j * np.sin(a)],
        [-1j * np.sin(a), np.cos(a)]
    ], dtype=complex)

    # Micromotion: exp(-i phi(L)/2 * sigma_z) in circular basis
    phi_L = m * np.sin(omega * L)
    U_micro = np.array([
        [np.exp(-1j * phi_L / 2), 0],
        [0, np.exp(+1j * phi_L / 2)]
    ], dtype=complex)

    # Spin-frame transformation: diag(e^{-i theta(L)}, e^{+i theta(L)})
    theta_L = (xi0 / omega) * np.sin(omega * L)
    U_spin = np.array([
        [np.exp(-1j * theta_L), 0],
        [0, np.exp(+1j * theta_L)]
    ], dtype=complex)

    # Full evolution in circular basis: U_spin @ U_micro @ U_secular
    J_circ = U_spin @ U_micro @ U_secular

    # Convert from circular (l,r) to linear (x,y) basis
    # T takes (x,y) → (l,r):  T = (1/√2) [[1, i], [1, -i]]
    # so J_xy = T^{-1} @ J_circ @ T
    T = np.array([[1, 1j], [1, -1j]]) / np.sqrt(2)
    Tinv = np.array([[1, 1], [-1j, 1j]]) / np.sqrt(2)
    J_xy = Tinv @ J_circ @ T

    info = {
        'm': m,
        'J0m': J0m,
        'beta_eff': beta_eff,
        'suppression': np.abs(J0m),
    }

    return J_xy, info


# =========================================================================
#  3.  Numerical subdivision — general spin profile
# =========================================================================
def calc_J_numerical(delta, xi_func, L, N=1000, alpha_circ=0.0):
    r"""
    Jones matrix by numerical subdivision for an arbitrary spin-rate
    profile :math:`\xi(z)`.

    Splits the fiber into N thin segments.  Each segment is modelled as
    a waveplate whose birefringence axes are rotated by the accumulated
    spin angle :math:`\theta(z) = \int_0^z \xi(z')\,dz'`.  The Jones
    matrices are multiplied together.

    Parameters
    ----------
    delta : float or callable
        Linear birefringence (rad/m).  If callable, ``delta(z)`` is
        evaluated at each segment midpoint.
    xi_func : callable
        Spin rate as a function of position, ``xi_func(z)`` (rad/m).
    L : float
        Total length (m).
    N : int, optional
        Number of subdivision segments.  Default 1000.
    alpha_circ : float or callable, optional
        Circular birefringence (rad/m).  Default 0.

    Returns
    -------
    J : np.ndarray, shape (2, 2), complex
        Jones matrix in the x-y linear basis.

    Notes
    -----
    This is the "brute force" approach discussed in Sec. 12.3 of the
    project notes.  For constant birefringence and sinusoidal spin,
    calc_J_sinusoidal is much faster when the Floquet conditions are met.

    The method works by noting that at position z the spin has rotated
    the birefringence axes by \theta(z).
    Circular birefringence is applied as an additional rotation.
    """
    dz = L / N
    J_total = np.eye(2, dtype=complex)

    # Accumulate the spin angle via trapezoidal integration
    theta = 0.0

    for n in range(N):
        z_mid = (n + 0.5) * dz

        # Local parameters
        delta_n = delta(z_mid) if callable(delta) else delta
        alpha_n = alpha_circ(z_mid) if callable(alpha_circ) else alpha_circ

        # Accumulated spin angle at segment midpoint
        theta_mid = theta + xi_func(z_mid) * dz / 2

        c, s = np.cos(theta_mid), np.sin(theta_mid)
        R_pos = np.array([[c, s], [-s, c]])    # R(theta)
        R_neg = np.array([[c, -s], [s, c]])    # R(-theta)

        phase = delta_n * dz / 2
        D = np.array([[np.exp(-1j * phase), 0],
                       [0, np.exp(+1j * phase)]], dtype=complex)

        J_seg = R_pos @ D @ R_neg

        # Circular birefringence: rotation by alpha*dz in x-y plane
        if alpha_n != 0:
            ca, sa = np.cos(alpha_n * dz / 2), np.sin(alpha_n * dz / 2)
            R_circ = np.array([[ca, -sa], [sa, ca]])
            J_seg = R_circ @ J_seg

        J_total = J_seg @ J_total

        theta += xi_func(z_mid) * dz

    return J_total


# =========================================================================
#  4.  Effective birefringence utilities
# =========================================================================
def effective_birefringence(delta, xi0, omega):
    r"""
    Effective (Floquet-renormalized) birefringence for sinusoidal spinning.

    .. math::
        \beta_{\rm eff} = \delta_0 \, J_0\!\left(\frac{2\xi_0}{\omega}\right)

    Parameters
    ----------
    delta : float
        Intrinsic linear birefringence (rad/m).
    xi0 : float
        Spin-rate amplitude (rad/m).
    omega : float
        Spatial angular frequency of the modulation (rad/m).

    Returns
    -------
    beta_eff : float
        Effective birefringence (rad/m).
    """
    m = 2 * xi0 / omega
    return delta * float(jv(0, m))


def suppression_factor(xi0, omega):
    r"""
    PMD suppression factor :math:`|J_0(2\xi_0/\omega)|`.

    At the zeros of :math:`J_0` the leading-order birefringence coupling
    vanishes.

    Parameters
    ----------
    xi0 : float
        Spin-rate amplitude (rad/m).
    omega : float
        Spatial angular frequency of the modulation (rad/m).

    Returns
    -------
    float
        |J_0(m)|
    """
    m = 2 * xi0 / omega
    return np.abs(float(jv(0, m)))


def optimal_modulation_indices(n_zeros=6):
    """
    Return the first n_zeros positive zeros of J_0(m),
    which are the modulation indices at which PMD suppression is
    maximized (to leading Floquet order).

    Parameters
    ----------
    n_zeros : int
        Number of zeros to return.

    Returns
    -------
    np.ndarray
        Array of length ``n_zeros`` with the zero locations.
    """
    from scipy.special import jn_zeros
    return jn_zeros(0, n_zeros)
