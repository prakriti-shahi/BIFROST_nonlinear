import numpy as np
import matplotlib.pyplot as plt
import fibers
import raman, raman_tabulated
raman_tabulated.install_tabulated_model(raman)

LAMBDA_Q  = 1550.52e-9
P_REF     = 1e-4
L_FIBER   = 26e3
T_K       = 293.15
DELTA_LAM = 0.1e-9

f = fibers.FiberLength(
    w0=LAMBDA_Q, T0=20.0, L0=L_FIBER,
    r0=4.1e-6, r1=62.5e-6, epsilon=1.005,
    m0=0.036, m1=0, Tref=20.0, rc=0, tf=0, tr=0
)

# Extend sweep to show the Raman peak and falloff
delta_nm = np.concatenate([
    np.linspace(0.5, 10, 40),
    np.linspace(10, 200, 100),
])

noise_stokes, noise_antistokes = [], []

for d in delta_nm:
    lam_ref_S  = LAMBDA_Q - d*1e-9   # ref bluer  → quantum channel is Stokes
    lam_ref_AS = LAMBDA_Q + d*1e-9   # ref redder → quantum channel is anti-Stokes
    r_S  = f.calcSpRamNoise(lam_ref_S,  LAMBDA_Q, DELTA_LAM, P_REF)
    r_AS = f.calcSpRamNoise(lam_ref_AS, LAMBDA_Q, DELTA_LAM, P_REF)
    noise_stokes.append(r_S['stokes_photons_per_sec'])
    noise_antistokes.append(r_AS['antistokes_photons_per_sec'])

# Convert to arrays and convert Raman shift axis to THz for the upper axis
noise_stokes     = np.array(noise_stokes)
noise_antistokes = np.array(noise_antistokes)
c = 299792458.0
raman_shift_THz = (c / (LAMBDA_Q - delta_nm*1e-9) - c/LAMBDA_Q) / 1e12

fig, ax = plt.subplots(figsize=(9, 5))

ax.semilogy(delta_nm, noise_stokes,     color='#1a6faf', lw=2,
            label='Stokes (ref bluer than λQ)')
ax.semilogy(delta_nm, noise_antistokes, color='#c0392b', lw=2, linestyle='--',
            label='Anti-Stokes (ref redder than λQ)')

# Mark the Raman peak position (~13.2 THz ≈ 105 nm at 1550 nm)
delta_peak_nm = (LAMBDA_Q*1e9)**2 / (c / 13.2e12 * 1e9)
ax.axvline(delta_peak_nm, color='gray', linestyle=':', lw=1.2, alpha=0.7)
ax.text(delta_peak_nm + 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 1e8, f'13.2 THz\nRaman peak\n(~{delta_peak_nm:.0f} nm)')

ax.set_xlabel('|λref − λQ|  (nm)', fontsize=11)
ax.set_ylabel('spRam noise  (photons/s in 100 pm BW)', fontsize=10)
ax.set_title('Spontaneous Raman Noise vs. WDM Channel Separation (BIFROST)\n'
             f'P = {P_REF*1e3:.1f} mW,  L = {L_FIBER/1e3:.0f} km,  '
             f'Δλ = {DELTA_LAM*1e9*1e3:.0f} pm,  T = {T_K:.0f} K  '
             f'[Lin & Agrawal tabulated model — zero beyond 25 THz]', fontsize=9.5)
ax.legend(fontsize=10)
ax.grid(True, which='both', linestyle='--', alpha=0.4)

# Upper x-axis showing Raman shift in THz
ax2 = ax.twiny()
# ax2.set_xlim(ax.get_xlim())
ax.set_xlim(0, 160)
tick_THz = [1, 2, 5, 10, 13.2, 20]
tick_nm  = [(LAMBDA_Q*1e9)**2 / (c/(t*1e12)*1e9) for t in tick_THz]
ax2.set_xticks(tick_nm)
ax2.set_xticklabels([f'{t}' for t in tick_THz], fontsize=8)
ax2.set_xlabel('Raman shift  Ω/(2π)  [THz]', fontsize=9)

plt.tight_layout()
plt.savefig('spRam_noise_vs_separation.pdf', bbox_inches='tight')
plt.savefig('spRam_noise_vs_separation.png', dpi=150, bbox_inches='tight')
plt.show()