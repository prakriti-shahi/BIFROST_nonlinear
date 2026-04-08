"""
spinning.py — Analytical Jones matrices for spun optical fibers.

Implements the spinning formalism from Sections 12.1–12.2 of the
Polarization Compensation Project Notes and the Floquet solution
for periodic spin rates (Britton, 3/13/2026).

Two regimes are supported:

1. **Constant Spin Rate, Constant Birefringence (CSRCB)**
   Closed-form Jones matrix via the matrix/Jones calculus methods
   (Eqns. 76/91 in the project notes; Eqn. 6 in Huang et al. 2024).

2. **Sinusoidal Spin Rate, Constant Birefringence (Floquet)**
   Analytical approximate Jones matrix valid when the modulation index
   m = 2*xi0/omega >> 1 and omega >= beta.  Uses the Jacobi–Anger /
   Bessel-function averaging result (Eqn. 21 in the Floquet notes).

Both functions return 2×2 complex Jones matrices in the x-y (linear)
basis.  They can be used as drop-in replacements for the diagonal
Jones matrix produced by ``_calc_J0`` in ``fibers.py`` whenever a
fiber segment is spun.

Patrick Banner & J. Britton
Dept of Physics — Swarthmore / UMD
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

    .. math::

        J_{\rm CSRCB}(L) = R(-\xi_0 L)
        \begin{pmatrix}
        \cos\gamma L - i\frac{\delta_0}{2\gamma}\sin\gamma L
          & \frac{\xi_0}{\gamma}\sin\gamma L \\
        -\frac{\xi_0}{\gamma}\sin\gamma L
          & \cos\gamma L + i\frac{\delta_0}{2\gamma}\sin\gamma L
        \end{pmatrix}

    where :math:`\gamma = \sqrt{(\delta_0/2)^2 + \xi_0^2}` (no circular
    birefringence) or
    :math:`\Omega/2 = \sqrt{(\beta/2)^2 + ((\alpha + 2\xi_0)/2)^2}`
    with circular birefringence included.

    Parameters
    ----------
    delta : float
        Intrinsic linear birefringence :math:`\delta_0 = \beta_y - \beta_x`
        (rad/m).  This is the total linear birefringence from all sources
        (CNC + ATS + BND) — **not** including twist, which is a circular
        birefringence and enters through ``alpha_circ``.
    xi0 : float
        Constant spin rate (rad/m).  Positive = counter-clockwise when
        looking along +z.
    L : float
        Length of the fiber segment (m).
    alpha_circ : float, optional
        Circular birefringence parameter :math:`\alpha = k_0 \epsilon_c / \bar n`
        (rad/m).  Default 0.

    Returns
    -------
    J : np.ndarray, shape (2, 2), complex
        Jones matrix in the x-y linear basis.

    Notes
    -----
    * When ``xi0 = 0`` and ``alpha_circ = 0`` the result reduces to the
      standard diagonal waveplate Jones matrix
      ``diag(exp(-i*delta*L/2), exp(+i*delta*L/2))``.
    * The spin enters the optics as a *rotation of the birefringence
      axes*, not as a twist-induced circular birefringence.  Physical
      twist birefringence (photoelastic shear) should be passed via
      ``alpha_circ``.
    """
    # Generalized Rabi-like frequency  (Eqn. 88 notation: Omega/2)
    Omega_half = np.sqrt((delta / 2) ** 2 + ((alpha_circ + 2 * xi0) / 2) ** 2)

    if Omega_half == 0:
        # No birefringence, no spin → identity
        return np.eye(2, dtype=complex)

    cosG = np.cos(Omega_half * L)
    sinG = np.sin(Omega_half * L)

    # Build the matrix in the rotating frame  (Eqn. 88, x-y basis)
    J_rot = np.array([
        [cosG - 1j * (delta / (2 * Omega_half)) * sinG,
         ((alpha_circ + 2 * xi0) / (2 * Omega_half)) * sinG],
        [-((alpha_circ + 2 * xi0) / (2 * Omega_half)) * sinG,
         cosG + 1j * (delta / (2 * Omega_half)) * sinG]
    ], dtype=complex)

    # Undo the rotating frame
    R = _rotation_matrix(-xi0 * L)
    return R @ J_rot


# =========================================================================
#  2.  Sinusoidal spin — Floquet / Bessel averaging
# =========================================================================
def calc_J_sinusoidal(delta, xi0, omega, L, alpha_circ=0.0,
                      n_bessel_terms=0):
    r"""
    Approximate Jones matrix for a fiber with sinusoidal spin rate
    :math:`\xi(z) = \xi_0 \cos(\omega z)` and constant linear
    birefringence :math:`\delta_0` (and optionally constant circular
    birefringence :math:`\alpha`).

    Valid in the Floquet regime :math:`m = 2\xi_0/\omega \gg 1` and
    :math:`\omega \gtrsim \delta_0`.

    The result is (Eqn. 21 of the Floquet notes):

    .. math::

        E(z) \approx
        \underbrace{\text{diag}(e^{-i\theta(z)}, e^{+i\theta(z)})}_{\text{spin frame}}
        \;
        \underbrace{\exp\!\bigl(-i\tfrac{\phi(z)}{2}\sigma_z\bigr)}_{\text{periodic micromotion}}
        \;
        \underbrace{\exp\!\bigl(-i\tfrac{\beta_{\rm eff}}{2}\sigma_x z\bigr)}_{\text{secular (Floquet)}}
        \; E(0)

    where :math:`\beta_{\rm eff} = \delta_0 \, J_0(m)` and
    :math:`\phi(z) = m\sin(\omega z)`,
    :math:`\theta(z) = (\xi_0/\omega)\sin(\omega z)`.

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
        When nonzero, the effective birefringence becomes
        :math:`\beta_{\rm eff} = \delta_0\,J_0(m)` and an effective
        detuning :math:`\alpha` remains; the secular evolution is then
        a CSRCB-like problem with these effective parameters.
    n_bessel_terms : int, optional
        Number of additional Bessel harmonics (beyond n=0) to include
        in the secular Hamiltonian.  Default 0 keeps only the leading
        :math:`J_0(m)` term.  Setting this to a positive integer includes
        corrections from the :math:`n = \pm 1, \ldots, \pm n` harmonics
        via second-order perturbation theory (Floquet–Magnus).
        **Not yet implemented** — reserved for future use.

    Returns
    -------
    J : np.ndarray, shape (2, 2), complex
        Jones matrix in the x-y linear basis.
    info : dict
        Dictionary with diagnostic quantities:

        * ``'m'`` — modulation index :math:`2\xi_0/\omega`
        * ``'J0m'`` — :math:`J_0(m)`
        * ``'beta_eff'`` — effective birefringence (rad/m)
        * ``'suppression'`` — :math:`|J_0(m)|`, the suppression factor

    Raises
    ------
    ValueError
        If ``omega <= 0``.

    Notes
    -----
    * At the zeros of :math:`J_0(m)` the leading-order effective coupling
      vanishes and intrinsic linear birefringence is strongly quenched.
    * The micromotion prefactors are periodic with period
      :math:`T = 2\pi/\omega` and average to the identity over many
      periods.  For fibers much longer than T, the secular part
      dominates.
    * For ``m < 1`` the approximation degrades; consider using the
      numerical CSRCB subdivision approach (``calc_J_numerical``)
      instead.
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

    # --- Secular (Floquet) evolution ---
    # In the interaction picture χ-frame (Eqn. 17–20), the secular
    # evolution is exp(-i * beta_eff/2 * sigma_x * L).
    # sigma_x in the circular basis corresponds to coupling between
    # the two circular modes.  In the x-y basis:
    #   exp(-i * a * sigma_x) = [[cos a, -i sin a], [-i sin a, cos a]]
    # but here sigma_x acts in the *circular* basis of Eqn. 17.
    #
    # To get the full Jones matrix in x-y, we need to:
    #   1. Apply the secular evolution in the χ-frame (circular basis)
    #   2. Undo the interaction-picture transformation (micromotion)
    #   3. Undo the rotating-frame transformation
    #   4. Convert back to x-y
    #
    # For the secular part alone (long-fiber / many-period limit where
    # micromotion averages out), the secular evolution in the circular
    # basis is:
    #   χ(L) ≈ [cos(β_eff L/2) I - i sin(β_eff L/2) σ_x] χ(0)
    #
    # To build the full matrix including micromotion at position L:

    a = beta_eff * L / 2   # half the accumulated effective phase

    # Secular evolution in the circular basis (Eqn. 20)
    U_secular = np.array([
        [np.cos(a), -1j * np.sin(a)],
        [-1j * np.sin(a), np.cos(a)]
    ], dtype=complex)

    # Micromotion: exp(-i phi(L)/2 * sigma_z) in circular basis
    # phi(L) = m * sin(omega * L)
    phi_L = m * np.sin(omega * L)
    U_micro = np.array([
        [np.exp(-1j * phi_L / 2), 0],
        [0, np.exp(+1j * phi_L / 2)]
    ], dtype=complex)

    # Spin-frame transformation: diag(e^{-i theta(L)}, e^{+i theta(L)})
    # theta(L) = integral of xi(z) dz = (xi0/omega) * sin(omega*L)
    #          = phi(L)/2 = (m/2) * sin(omega*L)
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

    Splits the fiber into N small CSRCB segments, each with spin rate
    :math:`\xi(z_n)` evaluated at the segment midpoint, and multiplies
    the Jones matrices together.

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
    ``calc_J_sinusoidal`` is much faster when the Floquet conditions
    are met.
    """
    dz = L / N
    J_total = np.eye(2, dtype=complex)

    # Accumulated spin angle for the rotation-frame bookkeeping
    # We track theta(z) = integral_0^z xi(z') dz'
    # and apply R(-theta(z_end)) @ J_local @ R(+theta(z_start))
    # But it's simpler to just use the CSRCB formula segment-by-segment,
    # noting that each segment has a local spin rate xi_n and the
    # accumulated spin angle only matters for the frame.
    #
    # Actually, for the CSRCB formula the spin rate is the *local* spin
    # rate of that segment and the rotation R(-xi0*L) at the end accounts
    # for the frame.  But when stitching segments together, we need
    # the cumulative spin angle.
    #
    # Correct approach: each small segment has birefringence axes rotated
    # by the accumulated spin angle theta_n.  The Jones matrix for a small
    # waveplate with axes at angle theta is:
    #   R(-theta) @ diag(e^{-i*delta*dz/2}, e^{+i*delta*dz/2}) @ R(theta)
    # where the spin *within* the segment causes additional rotation.
    #
    # For a small enough segment, the CSRCB formula with local xi is exact.
    # The key is that the frame rotation R(-xi0*dz) from each CSRCB
    # segment accumulates correctly when multiplied.

    theta = 0.0  # accumulated spin angle

    for n in range(N):
        z_mid = (n + 0.5) * dz
        z_start = n * dz

        # Local parameters
        xi_n = xi_func(z_mid)
        delta_n = delta(z_mid) if callable(delta) else delta
        alpha_n = alpha_circ(z_mid) if callable(alpha_circ) else alpha_circ

        # Small-segment Jones matrix using CSRCB
        # But we need to account for the fact that the birefringence
        # axes are rotated by the accumulated spin angle theta.
        #
        # The CSRCB formula already includes the local spinning within
        # the segment.  We just need to rotate the input/output by
        # the accumulated angle.
        J_seg = calc_J_CSRCB(delta_n, xi_n, dz, alpha_circ=alpha_n)

        # The CSRCB formula puts out the field in a frame rotated by
        # -xi_n*dz from the input.  To stitch correctly, we need to
        # account for the accumulated rotation of the birefringence axes.
        # Since CSRCB handles the *local* spin internally (including
        # R(-xi0*dz)), and each segment's birefringence axes start at
        # angle theta, we rotate:
        R_in = _rotation_matrix(theta)
        R_out = _rotation_matrix(-theta)

        J_total = R_out @ J_seg @ R_in @ J_total

        # Update accumulated spin angle
        theta += xi_n * dz

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
        :math:`|J_0(m)|` where :math:`m = 2\xi_0/\omega`.
    """
    m = 2 * xi0 / omega
    return np.abs(float(jv(0, m)))


def optimal_modulation_indices(n_zeros=5):
    """
    Return the first ``n_zeros`` positive zeros of :math:`J_0(m)`,
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
