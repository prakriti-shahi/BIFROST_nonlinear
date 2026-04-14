"""
deploy.py — Load fiber deployment configurations and compute link properties.

Reads a JSON config (or Python dict) describing a fiber route with multiple
sections and segments, creates the corresponding BIFROST objects, and provides
functions to compute Jones matrices, DGD, and noise budgets.

Segment types
-------------
  'straight'           Long run of fiber.  Spun if fiber-level spinning is
                       enabled.  → SpunFiberLength or FiberLength.
  'pole_wrap'          Fiber wrapped around a pole/drum.  Never spun (bending
                       dominates).  → FiberLength with rc, tf.
  'paddle_controller'  Manual polarization compensator.  Never spun (separate
                       device).  → FiberPaddleSet.
  'splice'             Fusion splice.  Inserts a Rotator hinge.  Tracks loss.
  'spun'               Explicit spun segment with custom spin parameters
                       (overrides fiber-level defaults).  → SpunFiberLength.

Hinge insertion
---------------
  Rotator hinges are inserted:
    - At section boundaries (underground→aerial, etc.)
    - At splice points
  Pole wraps act as deterministic "hinges" through their bending physics
  and do not get additional random rotators.
"""

import json
import numpy as np

# Try importing BIFROST under both possible names
try:
    import bifrost as bf
except ImportError:
    import fibers as bf

try:
    import spinning
except ImportError:
    spinning = None

try:
    import raman
except ImportError:
    raman = None

try:
    import brillouin
except ImportError:
    brillouin = None

pi = np.pi
C_c = 299792458.0


# ═══════════════════════════════════════════════════════════════════════════
#  Loading and building
# ═══════════════════════════════════════════════════════════════════════════

def load_deployment(config):
    """Load a deployment from a JSON file path or a Python dict.

    Parameters
    ----------
    config : str or dict
        If a string, interpreted as a path to a JSON file.
        If a dict, used directly as the config.

    Returns
    -------
    dict
        Deployment dict with keys:
        - ``'fibers'``: flat list of all BIFROST objects in order
        - ``'sections'``: list of section dicts, each with ``'name'``,
          ``'environment'``, ``'temperature_C'``, ``'objects'``,
          ``'splice_losses_dB'``
        - ``'raw_config'``: the original config dict
    """
    if isinstance(config, str):
        with open(config) as f:
            config = json.load(f)

    # --- Parse fiber-level properties ---
    fib = config['fiber']
    w0 = config['wavelength_nm'] * 1e-9
    r0 = fib['core_radius_um'] * 1e-6
    r1 = fib['cladding_radius_um'] * 1e-6
    m0 = fib['core_GeO2_mol_fraction']
    m1 = fib.get('cladding_GeO2_mol_fraction', 0.0)
    eps = fib.get('noncircularity_epsilon', 1.0)
    Tref = fib.get('reference_temperature_C', 20.0)

    # --- Parse spinning config ---
    spin_cfg = fib.get('spinning', {})
    spin_enabled = spin_cfg.get('enabled', False)
    spin_type = spin_cfg.get('spin_type', 'constant')
    spin_xi0 = spin_cfg.get('spin_rate_rad_per_m', 0.0)
    spin_omega = spin_cfg.get('spin_modulation_omega_rad_per_m', None)

    # --- Parse hinge model ---
    hinge_cfg = config.get('hinge_model', {})
    hinge_at_boundaries = hinge_cfg.get('at_section_boundaries', True)
    hinge_at_splices = hinge_cfg.get('at_splices', True)

    # --- Common kwargs for fiber objects ---
    def _common_kwargs(T0):
        return dict(w0=w0, r0=r0, r1=r1, epsilon=eps, m0=m0, m1=m1,
                    Tref=Tref, T0=T0)

    def _make_straight(seg, section_T):
        """Create a FiberLength or SpunFiberLength for a straight segment."""
        T0 = seg.get('temperature_C', section_T)
        L0 = seg['length_m']
        rc = seg.get('bend_radius_m', 0)
        tf = seg.get('tension_N', 0)
        tr = seg.get('twist_rate_rad_per_m', 0)

        # Check for per-segment spin override
        seg_spin = seg.get('spin_type', None)
        seg_xi0 = seg.get('spin_rate_rad_per_m', None)
        seg_omega = seg.get('spin_modulation_omega_rad_per_m', None)

        # Determine effective spin parameters
        use_spin = spin_enabled
        eff_type = spin_type
        eff_xi0 = spin_xi0
        eff_omega = spin_omega

        if seg_spin is not None:
            use_spin = True
            eff_type = seg_spin
        if seg_xi0 is not None:
            eff_xi0 = seg_xi0
        if seg_omega is not None:
            eff_omega = seg_omega

        if use_spin and eff_xi0 != 0:
            return bf.SpunFiberLength(
                w0=w0, T0=T0, L0=L0, r0=r0, r1=r1, epsilon=eps,
                m0=m0, m1=m1, Tref=Tref, rc=rc, tf=tf, tr=tr,
                xi0=eff_xi0, spin_type=eff_type, omega=eff_omega)
        else:
            return bf.FiberLength(
                w0=w0, T0=T0, L0=L0, r0=r0, r1=r1, epsilon=eps,
                m0=m0, m1=m1, Tref=Tref, rc=rc, tf=tf, tr=tr)

    def _make_pole_wrap(seg, section_T):
        """Create a FiberLength for a pole wrap (bending, never spun)."""
        T0 = seg.get('temperature_C', section_T)
        rc = seg['bend_radius_m']
        n_turns = seg.get('n_turns', 1)
        tf = seg.get('tension_N', 0)
        L0 = 2 * pi * rc * n_turns
        return bf.FiberLength(
            w0=w0, T0=T0, L0=L0, r0=r0, r1=r1, epsilon=eps,
            m0=m0, m1=m1, Tref=Tref, rc=rc, tf=tf, tr=0)

    def _make_paddle_controller(seg, section_T):
        """Create a FiberPaddleSet (never spun)."""
        T0 = seg.get('temperature_C', section_T)
        nP = seg['n_paddles']
        rps = np.array(seg['paddle_radii_m'])
        angles = np.array(seg['paddle_angles_deg']) * pi / 180
        Ns = np.array(seg['n_turns'])
        gapLs = np.array(seg['gap_lengths_m'])
        tfs = np.zeros(nP)
        return bf.FiberPaddleSet(
            w0=w0, T0=T0, r0=r0, r1=r1, epsilon=eps,
            m0=m0, m1=m1, Tref=Tref,
            nPaddles=nP, rps=rps, angles=angles,
            tfs=tfs, Ns=Ns, gapLs=gapLs)

    def _make_spun(seg, section_T):
        """Create a SpunFiberLength with explicit spin parameters."""
        T0 = seg.get('temperature_C', section_T)
        L0 = seg['length_m']
        rc = seg.get('bend_radius_m', 0)
        tf = seg.get('tension_N', 0)
        tr = seg.get('twist_rate_rad_per_m', 0)
        xi0 = seg['spin_rate_rad_per_m']
        st = seg.get('spin_type', 'constant')
        omega = seg.get('spin_modulation_omega_rad_per_m', None)
        return bf.SpunFiberLength(
            w0=w0, T0=T0, L0=L0, r0=r0, r1=r1, epsilon=eps,
            m0=m0, m1=m1, Tref=Tref, rc=rc, tf=tf, tr=tr,
            xi0=xi0, spin_type=st, omega=omega)

    # --- Build the deployment ---
    all_fibers = []
    sections_out = []

    for sec_idx, sec in enumerate(config['sections']):
        sec_name = sec.get('name', f'Section {sec_idx}')
        sec_env = sec.get('environment', 'unknown')
        # Section-level default temperature (fallback if segment doesn't specify)
        sec_T = sec.get('temperature_C', None)
        if sec_T is None:
            # Infer from first segment that has a temperature
            for seg in sec['segments']:
                if 'temperature_C' in seg:
                    sec_T = seg['temperature_C']
                    break
            if sec_T is None:
                sec_T = 20.0

        sec_objects = []
        sec_splices = []

        # Insert hinge at section boundary (except before the first section)
        if sec_idx > 0 and hinge_at_boundaries:
            rot = bf.makeRotators(1)[0]
            sec_objects.append(rot)

        for seg in sec['segments']:
            stype = seg['type']

            if stype == 'straight':
                sec_objects.append(_make_straight(seg, sec_T))

            elif stype == 'pole_wrap':
                sec_objects.append(_make_pole_wrap(seg, sec_T))

            elif stype == 'paddle_controller':
                sec_objects.append(_make_paddle_controller(seg, sec_T))

            elif stype == 'spun':
                sec_objects.append(_make_spun(seg, sec_T))

            elif stype == 'splice':
                loss_dB = seg.get('loss_dB', 0.05)
                sec_splices.append(loss_dB)
                # Insert a random rotator at the splice point
                if hinge_at_splices:
                    sec_objects.append(bf.makeRotators(1)[0])

            else:
                raise ValueError(f"Unknown segment type '{stype}' "
                                 f"in section '{sec_name}'")

        sections_out.append({
            'name': sec_name,
            'environment': sec_env,
            'temperature_C': sec_T,
            'objects': sec_objects,
            'splice_losses_dB': sec_splices,
        })
        all_fibers.extend(sec_objects)

    return {
        'fibers': all_fibers,
        'sections': sections_out,
        'raw_config': config,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Link-level computations
# ═══════════════════════════════════════════════════════════════════════════

def compute_jones(dep):
    """Total Jones matrix for the deployment (product of all elements)."""
    J = np.eye(2, dtype=complex)
    for f in dep['fibers']:
        J = f.J0 @ J
    return J


def compute_total_length(dep):
    """Total physical length of the deployment (m)."""
    return sum(f.L0 for f in dep['fibers'])


def compute_dgd(dep, dw0=0.1e-9):
    """DGD of the full deployment via Jones matrix eigenanalysis (s).

    Parameters
    ----------
    dep : dict
        Deployment from ``load_deployment``.
    dw0 : float
        Wavelength step for numerical differentiation (m).

    Returns
    -------
    float
        DGD in seconds.
    """
    w0 = dep['raw_config']['wavelength_nm'] * 1e-9

    def _full_jones_at(wavelength):
        """Recompute full Jones matrix at a different wavelength."""
        # Temporarily change wavelength on all objects
        old_w0s = []
        for f in dep['fibers']:
            old_w0s.append(getattr(f, 'w0', None))
            if hasattr(f, 'w0'):
                f.w0 = wavelength
        J = compute_jones(dep)
        # Restore
        for f, ow in zip(dep['fibers'], old_w0s):
            if ow is not None:
                f.w0 = ow
        return J

    Jb = compute_jones(dep)
    Ja = _full_jones_at(w0 - dw0)
    Jc = _full_jones_at(w0 + dw0)

    # Restore wavelength
    for f in dep['fibers']:
        if hasattr(f, 'w0'):
            f.w0 = w0

    # Eigenanalysis for ± dw0
    matM = Ja @ np.linalg.inv(Jb)
    valsM = np.linalg.eigvals(matM)
    dgdM = np.abs(np.angle(valsM[0] / valsM[1]) /
                   ((2 * pi * C_c) / w0**2 * dw0))

    matP = Jc @ np.linalg.inv(Jb)
    valsP = np.linalg.eigvals(matP)
    dgdP = np.abs(np.angle(valsP[0] / valsP[1]) /
                   ((2 * pi * C_c) / w0**2 * dw0))

    return (dgdM + dgdP) / 2


def compute_noise_budget(dep):
    """Per-segment Raman and Brillouin noise budget.

    Returns
    -------
    dict
        Keys: ``'pump_wavelength_nm'``, ``'pump_power_mW'``,
        ``'quantum_wavelength_nm'``, ``'segments'`` (list of per-segment dicts),
        ``'total_raman_photons_per_s'``, ``'total_brillouin_photons_per_s'``.
    """
    cfg = dep['raw_config']
    analysis = cfg.get('analysis', {})
    lam_pump = analysis.get('pump_wavelength_nm', 1550) * 1e-9
    P_pump = analysis.get('pump_power_mW', 1.0) * 1e-3
    lam_q = analysis.get('quantum_channel_wavelength_nm', 1310) * 1e-9
    dlam_q = analysis.get('quantum_channel_bandwidth_nm', 1.0) * 1e-9

    segments = []
    total_raman = 0.0
    total_bril = 0.0

    for i, f in enumerate(dep['fibers']):
        seg_info = {
            'index': i,
            'type': type(f).__name__,
            'L0': f.L0,
            'raman_photons_per_s': 0.0,
            'brillouin_photons_per_s': 0.0,
            'sbs_threshold_mW': None,
        }

        if hasattr(f, 'calcSpRamNoise') and f.L0 > 0:
            try:
                ram = f.calcSpRamNoise(lam_pump, lam_q, dlam_q, P_pump)
                rph = ram.get('stokes_photons_per_sec', 0.0)
                seg_info['raman_photons_per_s'] = rph
                total_raman += rph
            except Exception:
                pass

        if hasattr(f, 'calcSpBrilNoise') and f.L0 > 0:
            try:
                bril = f.calcSpBrilNoise(lam_pump, lam_q, dlam_q, P_pump)
                bph = bril.get('backward_photon_rate', 0.0)
                seg_info['brillouin_photons_per_s'] = bph
                total_bril += bph
            except Exception:
                pass

        if hasattr(f, 'brillouinThreshold') and f.L0 > 0:
            try:
                th = f.brillouinThreshold(lambda_ref=lam_pump)
                seg_info['sbs_threshold_mW'] = th['P_threshold_mW']
            except Exception:
                pass

        segments.append(seg_info)

    return {
        'pump_wavelength_nm': lam_pump * 1e9,
        'pump_power_mW': P_pump * 1e3,
        'quantum_wavelength_nm': lam_q * 1e9,
        'segments': segments,
        'total_raman_photons_per_s': total_raman,
        'total_brillouin_photons_per_s': total_bril,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience summaries
# ═══════════════════════════════════════════════════════════════════════════

def print_deployment_summary(dep):
    """Print a human-readable summary of the deployment."""
    print(f"{'Section':40s}  {'Env':12s}  {'T (°C)':>6}  "
          f"{'Objects':>7}  {'Length (m)':>10}")
    print('-' * 82)
    for sec in dep['sections']:
        n = len(sec['objects'])
        L = sum(f.L0 for f in sec['objects'])
        splices = sec['splice_losses_dB']
        sp_str = f"  splices: {splices}" if splices else ""
        print(f"  {sec['name']:38s}  {sec['environment']:12s}  "
              f"{sec['temperature_C']:>6.1f}  {n:>7d}  {L:>10.2f}{sp_str}")
    print(f"\n  Total length: {compute_total_length(dep):.1f} m")
    print(f"  Total objects: {len(dep['fibers'])}")

    # Count types
    types = {}
    for f in dep['fibers']:
        t = type(f).__name__
        types[t] = types.get(t, 0) + 1
    print(f"  Object types: {types}")


def print_segment_table(dep):
    """Print per-segment properties table."""
    print(f"{'#':>3}  {'Type':20s}  {'L (m)':>8}  "
          f"{'Spin':>10}  {'T (°C)':>6}")
    print('-' * 60)
    for i, f in enumerate(dep['fibers']):
        name = type(f).__name__
        L = f.L0
        spin_str = '—'
        T_str = '—'

        if hasattr(f, 'spin_type'):
            if f.spin_type == 'sinusoidal':
                m = 2 * f.xi0 / f.omega if f.omega else 0
                spin_str = f'sin m={m:.0f}'
            else:
                spin_str = f'const {f.xi0:.0f}'
        elif hasattr(f, 'tr') and hasattr(f, 'rc'):
            if f.rc != 0:
                spin_str = f'bend r={f.rc:.2f}'

        if hasattr(f, 'T0'):
            T_str = f'{f.T0:.1f}'

        print(f"{i:>3}  {name:20s}  {L:>8.3f}  "
              f"{spin_str:>10}  {T_str:>6}")
