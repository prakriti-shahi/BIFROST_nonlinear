"""
plot_raman_spectrum.py — Visual validation of the Blow-Wood g_R(Ω) spectrum.

Compares the BIFROST raman.py implementation against the key features of
Figure 8.2 in Agrawal, *Nonlinear Fiber Optics*, 5th/6th ed., which itself
reproduces the original Stolen & Ippen (1973) measurement.

Run from the directory containing raman.py:
    python plot_raman_spectrum.py

Saves:  raman_spectrum_validation.pdf  (and .png)

What to look for
----------------
The Blow-Wood model is a two-parameter Lorentzian (τ₁, τ₂).  It should match:
  ✓  Peak near Ω/(2π) ≈ 13.2 THz
  ✓  Smooth near-zero behaviour below ~3 THz
  ✗  Secondary shoulder near 15 THz  — Blow-Wood cannot reproduce this
  ✗  Rapid falloff to zero beyond ~20–25 THz — Blow-Wood falls off too slowly

Features the model misses are annotated on the plot so the comparison is honest.

References
----------
Agrawal, *Nonlinear Fiber Optics*, 5th ed. (2013), Fig. 8.2.
Stolen & Ippen, *Appl. Phys. Lett.* 22, 276 (1973).
Blow & Wood, *IEEE J. Quantum Electron.* 25, 2665 (1989).
Lin & Agrawal, *Opt. Lett.* 31, 3086 (2006) — tabulated spectrum for upgrade.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings

try:
    import raman
except ImportError:
    raise ImportError("Cannot import raman.py.  Run from its directory.")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
pi    = np.pi
c     = 299_792_458.0      # m/s
gamma = 1.3e-3             # W⁻¹m⁻¹ — SMF-28 at 1550 nm

# Frequency axis: 0 → 40 THz (show the full Raman band plus tail)
f_THz   = np.linspace(0.0, 40.0, 8001)         # THz
Omega   = f_THz * 1e12 * 2*pi                  # rad/s

# Compute g_R — suppress the V-number warning if fibers also imported
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    gR = np.array([float(np.atleast_1d(raman.g_R(np.array([Om]), gamma))[0])
                   for Om in Omega])

# Normalise to peak = 1 for shape comparison (Agrawal Fig 8.2 is normalised)
gR_peak_val  = gR.max()
f_peak       = float(f_THz[np.argmax(gR)])
gR_norm      = gR / gR_peak_val

# ---------------------------------------------------------------------------
# Digitised reference points from Agrawal Fig 8.2 / Stolen & Ippen
# These are approximate, read off the published figure.
# Values are (frequency [THz], normalised g_R).
# ---------------------------------------------------------------------------
stolen_ippen = np.array([
    [0.0,  0.000],
    [1.0,  0.020],
    [2.0,  0.060],
    [3.0,  0.120],
    [4.0,  0.185],
    [5.0,  0.260],
    [6.0,  0.345],
    [7.0,  0.430],
    [8.0,  0.530],
    [9.0,  0.640],
    [10.0, 0.745],
    [11.0, 0.840],
    [12.0, 0.920],
    [13.2, 1.000],   # primary peak
    [14.5, 0.960],   # shoulder begins
    [15.0, 0.930],   # secondary shoulder (≈ 15 THz) — NOT in Blow-Wood
    [16.0, 0.820],
    [17.0, 0.640],
    [18.0, 0.430],
    [19.0, 0.250],
    [20.0, 0.140],
    [21.0, 0.075],
    [22.0, 0.038],
    [23.0, 0.018],
    [25.0, 0.005],
    [28.0, 0.000],
])

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 9),
                          gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.38})

ax   = axes[0]   # main spectrum comparison
ax_r = axes[1]   # residuals / difference panel

fig.patch.set_facecolor('#fafafa')
for a in axes:
    a.set_facecolor('#fafafa')

# ---------------------------------------------------------------------------
# Main panel — normalised spectra
# ---------------------------------------------------------------------------
# Blow-Wood model
ax.plot(f_THz, gR_norm,
        color='#1a6faf', linewidth=2.2, label='Blow–Wood model (BIFROST raman.py)',
        zorder=3)

# Stolen & Ippen / Agrawal reference
ax.plot(stolen_ippen[:, 0], stolen_ippen[:, 1],
        color='#c0392b', linewidth=0.0,
        marker='o', markersize=5.5, markerfacecolor='#c0392b',
        markeredgewidth=0.8, markeredgecolor='#800000',
        label='Stolen & Ippen (1973) / Agrawal Fig. 8.2\n(digitised reference)',
        zorder=4)

# Shade the region where Blow-Wood is expected to agree (< ~18 THz)
ax.axvspan(0, 18, alpha=0.06, color='#1a6faf', zorder=0)
ax.axvspan(18, 40, alpha=0.06, color='#c0392b', zorder=0)

# Annotation: primary peak
ax.annotate(f'Primary peak\n{f_peak:.1f} THz',
            xy=(f_peak, 1.0), xytext=(f_peak + 3.5, 0.97),
            fontsize=9, color='#1a6faf',
            arrowprops=dict(arrowstyle='->', color='#1a6faf', lw=1.2),
            ha='left')

# Annotation: secondary shoulder (reference only)
ax.annotate('Secondary shoulder\n~15 THz\n(Blow–Wood misses this)',
            xy=(15.0, 0.930), xytext=(20.0, 0.88),
            fontsize=8.5, color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2),
            ha='left')

# Annotation: slow tail
ax.annotate('Blow–Wood tail\ntoo slow here',
            xy=(28.0, float(gR_norm[np.searchsorted(f_THz, 28.0)])),
            xytext=(31.0, 0.25),
            fontsize=8.5, color='#1a6faf',
            arrowprops=dict(arrowstyle='->', color='#1a6faf', lw=1.2),
            ha='left')

# Annotation: near-zero region
ax.annotate('Near-zero\nbelow ~3 THz\n✓ both agree',
            xy=(2.5, float(gR_norm[np.searchsorted(f_THz, 2.5)])),
            xytext=(5.5, 0.35),
            fontsize=8.5, color='#2ecc71',
            arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.2),
            ha='left')

# Region labels
ax.text(8.5, 0.05, 'Blow–Wood\nreliable region', fontsize=8,
        color='#1a6faf', alpha=0.7, ha='center')
ax.text(28.0, 0.05, 'Blow–Wood\noverestimates tail', fontsize=8,
        color='#c0392b', alpha=0.7, ha='center')

ax.set_xlim(0, 40)
ax.set_ylim(-0.05, 1.18)
ax.set_xlabel('Raman shift  Ω/(2π)  [THz]', fontsize=11)
ax.set_ylabel('Normalised Raman gain  $g_R / g_{R,\\mathrm{peak}}$', fontsize=11)
ax.set_title(
    'Raman Gain Spectrum Validation — Blow–Wood vs. Stolen & Ippen (1973)\n'
    r'$g_R(\Omega) = 2\gamma f_R \, \mathrm{Im}[\tilde{h}_R(\Omega)]$  '
    rf'with  $\gamma = {gamma*1e3:.1f}$ W⁻¹km⁻¹  (SMF-28, 1550 nm)',
    fontsize=11, pad=10)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

# Model parameter box
param_text = (
    f'Blow–Wood parameters\n'
    f'  τ₁ = {raman.RAMAN_TAU1_SI*1e15:.1f} fs  →  peak at {1/(raman.RAMAN_TAU1_SI*1e12*2*pi):.1f} THz\n'
    f'  τ₂ = {raman.RAMAN_TAU2_SI*1e15:.0f} fs  →  linewidth\n'
    f'  f_R = {raman.RAMAN_FR_SI:.2f}\n'
    f'  $g_{{R,\\mathrm{{peak}}}}$ = {gR_peak_val:.3e} W⁻¹m⁻¹'
)
ax.text(0.02, 0.97, param_text, transform=ax.transAxes,
        fontsize=8.2, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='#aaaaaa', alpha=0.9))

# ---------------------------------------------------------------------------
# Residuals panel — absolute difference (on the region where we have ref data)
# ---------------------------------------------------------------------------
# Interpolate Blow-Wood onto the reference frequency grid
from numpy import interp
f_ref    = stolen_ippen[:, 0]
gR_ref   = stolen_ippen[:, 1]
gR_bw_at_ref = interp(f_ref, f_THz, gR_norm)

residual = gR_bw_at_ref - gR_ref

ax_r.bar(f_ref, residual,
         width=0.6, color=np.where(residual >= 0, '#1a6faf', '#c0392b'),
         alpha=0.75, zorder=3, label='Blow–Wood − reference')
ax_r.axhline(0, color='#333333', linewidth=0.8)
ax_r.axhline(+0.15, color='#888888', linewidth=0.6, linestyle='--')
ax_r.axhline(-0.15, color='#888888', linewidth=0.6, linestyle='--')
ax_r.text(30, 0.17, '±0.15 guide', fontsize=7.5, color='#888888')

ax_r.set_xlim(0, 40)
ax_r.set_ylim(-0.45, 0.55)
ax_r.set_xlabel('Raman shift  Ω/(2π)  [THz]', fontsize=11)
ax_r.set_ylabel('Residual\n(B–W minus ref)', fontsize=9.5)
ax_r.set_title('Residuals: Blow–Wood model minus digitised reference',
               fontsize=10)
ax_r.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

# Label the shoulder region in residuals
ax_r.annotate('shoulder missed\nby Blow–Wood',
              xy=(15.0, gR_bw_at_ref[np.searchsorted(f_ref, 15.0)] - 0.930),
              xytext=(18.0, -0.30),
              fontsize=7.5, color='#c0392b',
              arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.9))

# ---------------------------------------------------------------------------
# Feature checklist printed in figure
# ---------------------------------------------------------------------------
checklist = (
    "Feature checklist (Agrawal Fig. 8.2)\n"
    "──────────────────────────────────────\n"
    f"  Peak ≈ 13.2 THz       {'✓' if 12.5 < f_peak < 14.0 else '✗'}  ({f_peak:.1f} THz)\n"
    "  Shoulder ≈ 15 THz     ✗  (Blow–Wood is smooth)\n"
    f"  Near-zero below 3 THz {'✓' if gR_norm[np.searchsorted(f_THz, 3.0)] < 0.15 else '✗'}\n"
    f"  Falloff beyond 20 THz ✗  (model tail too slow)\n"
    "──────────────────────────────────────\n"
    "  → Upgrade: Lin & Agrawal (2006)\n"
    "    tabulated h_R for accurate tails"
)
ax_r.text(1.02, 1.35, checklist, transform=ax_r.transAxes,
          fontsize=7.8, verticalalignment='top', family='monospace',
          bbox=dict(boxstyle='round,pad=0.6', facecolor='#fffef0',
                    edgecolor='#cccc88', alpha=0.95))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
for fmt in ('pdf', 'png'):
    fname = f'raman_spectrum_validation.{fmt}'
    fig.savefig(fname, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved: {fname}")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print("Raman spectrum feature check (Agrawal Fig. 8.2)")
print("=" * 55)
checks = [
    ("Peak ≈ 13.2 THz",        12.5 < f_peak < 14.0,
     f"peak at {f_peak:.2f} THz"),
    ("Near-zero below 3 THz",
     float(gR_norm[np.searchsorted(f_THz, 3.0)]) < 0.15,
     f"g_R(3 THz)/g_R_peak = {gR_norm[np.searchsorted(f_THz, 3.0)]:.3f}"),
    ("Secondary shoulder ~15 THz", False,
     "Blow–Wood cannot reproduce this (two-param Lorentzian)"),
    ("Rapid falloff beyond 20 THz", False,
     f"g_R(25 THz)/g_R_peak = {gR_norm[np.searchsorted(f_THz, 25.0)]:.3f} "
     f"(should be ~0)"),
]
for name, passed, detail in checks:
    mark = "✓ PASS" if passed else "✗ NOTE"
    print(f"  [{mark}]  {name}")
    print(f"           {detail}")
print()
print("  The two NOTE items are expected Blow–Wood model limitations.")
print("  Upgrade path: tabulated h_R from Lin & Agrawal, Opt. Lett. 31, 3086 (2006).")
print("=" * 55)