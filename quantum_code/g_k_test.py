"""Run deze code om de 3D harmonische oscillator en verstrooiingsprobleem op te lossen."""

# Naam: Monsieurs Pieter
# E-mail: s0224946@ad.ua.ac.be
# Solo, met AI-assistentie
# Title: Testprogramma voor gevorderde kwantum taak DONE PIETER

# Importeer de volgende bibliotheken
import numpy as np  # om roosterpunten aan te maken.
import matplotlib.pyplot as plt  # om een rudimentaire plot te maken.
from scipy.optimize import fsolve  # voor optimalisatie en "root finding" (voor generaliseerd numerov)
import scipy.signal as signal  # Om pieken te vinden.

# Importeer alle nodige functies uit het bestand g_k_main.py
# Ik importeer ze nu allemaal afzonderlijk. Dit werkt, maar ik had ze ook 
# in aparte klassen kunnen zetten, en om dan die klassen te importeren.
# ... moving on.
from g_k_main import (
    matrix_numerov, matrix_numerov_cartesian_1d, generalized_numerov_prof_forward,
    generalized_numerov_prof_backward, compute_cross_section_method_with_printing,
    born_scattering_cross_section_multiwav, compute_born_phase_shifts_with_printing,
    effective_potential, F_scattering, EPSILON, SIGMA, FACTOR_2M_HBAR2, CONV_ANG_SQ
)

# Dit is hoe je een aanroeping start.
if __name__ == "__main__":
    # Het kan soms verwarrend zijn van welke run we zijn, dus
    # dit tekstje zou moeten helpen.
    print("\n===============================")
    print(" Hello world, this is a new run")
    print("================================\n")

    # We maken ons rooster nu aan.
    # We stellen het zodanig in, dat het eerste roosterpunt
    # onze stapgrootte h is, zoals geadviseerd door de prof.
    n_points = 700
    r_max = 5.5
    h = r_max / n_points
    r_grid = np.linspace(h, r_max, n_points)
    
    x_max = 5.5
    x_cart_grid = np.linspace(-x_max, x_max, n_points)
    
    # De effectieve potentiaal
    # Voor l = 0, en l = 1.
    V_eff_0 = 0.5 * r_grid**2
    V_eff_1 = 0.5 * r_grid**2 + 1.0 / (r_grid**2)
    V_1d = 0.5 * x_cart_grid**2
    
    # Dit berekent de energieniveaus voor de drie methoden:
    # Generaliseerd Numerov, matrix Numerov sferisch, en matrix Numerov cartesisch,  
    E_mat_l0, _ = matrix_numerov(r_grid, V_eff_0)
    E_mat_l1, _ = matrix_numerov(r_grid, V_eff_1)
    E_1d, _ = matrix_numerov_cartesian_1d(x_cart_grid, V_1d)
    
    # Dit is de basis info voor elk energieniveau, waaronder de
    # meeste belangrijke zaken zijn:
    # l, n_r, de theoretische waarde, en de gok die we meegeven.
    states_info = [
        ("E0 (l=0, nr=0)", 0, 0, 1.5, 1.48, 1.73, 0),
        ("E1 (l=1, nr=0)", 1, 0, 2.5, 2.48, 2.00, 1),
        ("E2 (l=0, nr=1)", 0, 1, 3.5, 3.48, 1.73, 2),
        ("E3 (l=1, nr=1)", 1, 1, 4.5, 4.48, 2.00, 3),
        ("E4 (l=0, nr=2)", 0, 2, 5.5, 5.48, 1.73, 4),
        ("E5 (l=1, nr=2)", 1, 2, 6.5, 6.48, 2.00, 5),
    ]
    
    # Dit gaat een mastertabel zijn, dus we maken het bombastisch.
    print("===================================================================================================================")
    print(f"{'Toestand':<16} | {'Theorie':<8} | {'Mat. Sferisch':<13} | {'|dE| Sferisch':<13} | {'Mat. Cartes':<13} | {'|dE| Cartes':<13} | {'Prof Gen.':<12} | {'|dE| Prof':<11}")
    print("===================================================================================================================")
    
    # We gaan ook de absolute verschillen berekenen, dus we 
    # bereiden even een paar lege lijsten voor, omdat we straks 
    # hen gaan vullen.
    err_matrix_sph, err_matrix_cart, err_prof = [], [], []
    labels = []

    # loop in essentie over elk tupel uit de lijst "states_info".
    for label, l, n_r, E_th, guess, r_m, cart_n in states_info:
    
        # We voegen het label toe aan de labelslijst.
        labels.append(label)
        
        # Dit berekent het absoluut verschil voor sferisch matrix numerove.
        E_sph = E_mat_l0[n_r] if l == 0 else E_mat_l1[n_r]
        err_sph = abs(E_sph - E_th)
        err_matrix_sph.append(err_sph)
        
        # Dit berekent het absoluut verschil voor cartesisch matrix numerove.
        E_cart = E_1d[cart_n] + 2.0 * E_1d[0]
        err_cart = abs(E_cart - E_th)
        err_matrix_cart.append(err_cart)
        
        # Dit berekent het absoluut verschil voor generaliseerd numerove.
        # Omdat deze methode bestaat uit zowel voorwaarts als achterwaarts, moeten we 
        # hen eerst nog "aan elkaar lijmen".
        V_eff = V_eff_0 if l == 0 else V_eff_1
        match_idx = np.searchsorted(r_grid, r_m)
        
        def prof_err(E_arr):
            """"Bereken het hellingsverschil op het matchingpunt om gebonden toestanden te vinden via fsolve."""
            E_val = E_arr[0]
            F_arr = 2.0 * (V_eff - E_val)
            u0, u1 = r_grid[0]**(l + 1), r_grid[1]**(l + 1)
            u_f = generalized_numerov_prof_forward(r_grid[:match_idx+2], F_arr[:match_idx+2], u0, u1)
            u_b = generalized_numerov_prof_backward(r_grid[match_idx-1:], F_arr[match_idx-1:], 0.0, 1e-10)
            u_b *= (u_f[match_idx] / u_b[1])
            d_f = (u_f[match_idx+1] - u_f[match_idx-1]) / (2.0 * h)
            d_b = (u_b[2] - u_b[0]) / (2.0 * h)
            return [d_f - d_b]
        
        # Dit geeft het uieindelijke antwoord van DE generaliseerde numerov methode.
        E_prof = fsolve(prof_err, [guess])[0]

        # Dit berekent dat het absoluut verschil.
        err_p = abs(E_prof - E_th)
        err_prof.append(err_p)
        
        # We printen alle resultaten op het scherm.
        print(f"{label:<16} | {E_th:<8.1f} | {E_sph:<13.6f} | {err_sph:<13.2e} | {E_cart:<13.6f} | {err_cart:<13.2e} | {E_prof:<12.6f} | {err_p:<11.2e}")
    
    # De tabel is gedaan.
    print("==================================================================================================================-\n")

    # Nu maken we nog een histogram om alle absolute verschillen van elke methode met elkaar te vergelijken.
    # Nothing too fancy here.
    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 6))
    rects1 = ax.bar(x - width, err_matrix_sph, width, label='Matrix Numerov (Sferisch)', color='steelblue', edgecolor='black')
    rects2 = ax.bar(x, err_matrix_cart, width, label='Matrix Numerov (Cartesisch)', color='forestgreen', edgecolor='black')
    rects3 = ax.bar(x + width, err_prof, width, label="Prof's Generalized Numerov", color='darkorange', edgecolor='black')

    ax.set_yscale('log')
    ax.set_ylabel(r"Absolute fout $|E_{\mathrm{num}} - E_{\mathrm{theorie}}|$", fontsize=11)
    ax.set_title("Nauwkeurigheid: Matrix (Sferisch) vs. Matrix (Cartesisch) vs. Prof Numerov ($E_0$ t/m $E_5$)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(1e-13, 1e-3)
    ax.grid(True, axis='y', which='both', linestyle='--', alpha=0.35)
    ax.legend(fontsize=10)

    # Getallen netjes boven elke balk ook gezet
    # Omdat het mooi is, en om de resultaten te checken.
    for rect in rects1:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}", (rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, color='navy')
    for rect in rects2:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}", (rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, color='darkgreen')
    for rect in rects3:
        val = rect.get_height()
        ax.annotate(f"{val:.1e}", (rect.get_x() + rect.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, color='darkred')

    plt.tight_layout()
    plt.show()

    # Nu printen we nog de faseverschuivingen zelf
    # Die vormen immers HET antwoord op het verstrooiingsprobleem.
    E_vals = np.linspace(0.1, 3.5, 300)
    l_max = 6
    
    # Numerov faseverschuivingen printen
    _, _, sigma_gen, partial_gen = compute_cross_section_method_with_printing('generalized', E_vals, l_max=l_max)
    sigma_born = born_scattering_cross_section_multiwav(E_vals, l_max=l_max)

    # Born faseverschuivingen printen.
    target_print_energies = [0.1, 1.237, 2.374]
    compute_born_phase_shifts_with_printing(target_print_energies, l_max=l_max)

    # We zoeken nu ook nog de pieken.
    # Eerst alle pieken.
    peaks_gen, _ = signal.find_peaks(sigma_gen, prominence=5.0)
    peaks_born, _ = signal.find_peaks(sigma_born, prominence=5.0)

    # Print nu alle pieken.
    print("\n--- ALLE GEVONDEN PIEKEN (NUMEROV) ---")
    for idx in peaks_gen:
        print(f"E = {E_vals[idx]:.4f} meV | sigma = {sigma_gen[idx]:.2f} Ã…Â²")

    print("\n--- ALLE GEVONDEN PIEKEN (BORN) ---")
    for idx in peaks_born:
        print(f"E = {E_vals[idx]:.4f} meV | sigma = {sigma_born[idx]:.2f} Ã…Â²")

    # De literatuur verwacht drie pieken, dus laten we ons beperken tot de top 3.
    top3_gen = peaks_gen[np.argsort(sigma_gen[peaks_gen])[-3:]]
    top3_born = peaks_born[np.argsort(sigma_born[peaks_born])[-3:]]

    # En nu printen we die top 3 voor elke methode.
    print("\n--- TOP 3 HOOGSTE PIEKEN (NUMEROV) ---")
    for idx in top3_gen[::-1]:
        print(f"E = {E_vals[idx]:.4f} meV | sigma = {sigma_gen[idx]:.2f} Ã…Â²")

    print("\n--- TOP 3 HOOGSTE PIEKEN (BORN) ---")
    for idx in top3_born[::-1]:
        print(f"E = {E_vals[idx]:.4f} meV | sigma = {sigma_born[idx]:.2f} Ã…Â²")

    # We hebben voor een verdere analyse ook nog de breedtes van de pieken nodig
    # waaruit we dan de levensduren halen. Dat komt nu.
    print("\n--- PIEKBREEDTES (FWHM / GAMMA) VOOR TOP 3 PIEKEN ---")
    
    # We loopen over de top 3 pieken van de generaliseerde Numerov methode.
    for idx in top3_gen:
        
        # We passen nu het begrip voor FWHM toe.
        # Full width = volle breedte
        # Half height = halve hoogte, dus peak_sigma/2
        peak_sigma = sigma_gen[idx]
        half_max = peak_sigma / 2.0
        
        # Voor de linkerkant
        left_side = sigma_gen[:idx]
        E_left_grid = E_vals[:idx]
        idx_l = np.argmin(np.abs(left_side - half_max)) if len(left_side) > 0 else idx
        E1 = E_left_grid[idx_l]
        
        # En de rechterkant
        right_side = sigma_gen[idx:]
        E_right_grid = E_vals[idx:]
        idx_r = np.argmin(np.abs(right_side - half_max)) if len(right_side) > 0 else 0
        E2 = E_right_grid[idx_r]
        
        # De breedte is dan gewoonweg het verschil tussen die twee energieen.
        gamma = E2 - E1
        
        # We printen het resultaat op het scherm.
        print(f"Piek bij E = {E_vals[idx]:.4f} meV -> Breedte (Gamma) = {gamma:.4f} meV")

    # En dan ook nog dezelfde behandeling voor de partiele golven zelf.
    # De werkwijze is analoog.
    print("\n" + "="*70)
    print(f"{'Impulsmoment (l)':<18} | {'FWHM Breedte Î“ (meV)':<20}")
    print("="*70)
    
    for l in range(l_max + 1):
        sig_l = partial_gen[l]
        idx_max = np.argmax(sig_l)
        peak_sigma = sig_l[idx_max]
        half_max = peak_sigma / 2.0
        left_side = sig_l[:idx_max]
        E_l_grid = E_vals[:idx_max]
        E1 = E_l_grid[np.argmin(np.abs(left_side - half_max))] if len(left_side) > 0 else E_vals[idx_max]
        right_side = sig_l[idx_max:]
        E_r_grid = E_vals[idx_max:]
        E2 = E_r_grid[np.argmin(np.abs(right_side - half_max))] if len(right_side) > 0 else E_vals[idx_max]
        gamma = E2 - E1
        print(f"l = {l:<15} | {gamma:<20.4f}")
    print("="*70 + "\n")

    # Nu plotten we de curves van de partiele golven, sigma_tot, en 
    # de top drie pieken daarbovenop natuurlijk.
    plt.figure(figsize=(10, 6))
    plt.plot(E_vals, sigma_gen, 'k-', linewidth=2.5, label=r"$\sigma_{\text{tot}}$ (Numerov)")
    plt.plot(E_vals, sigma_born, 'm:', linewidth=2.0, label="Born-benadering")
    
    # Plot Numerov en Born top 3 pieken
    plt.plot(E_vals[top3_gen], sigma_gen[top3_gen], 'ro', markersize=8, zorder=5, label="Top 3 Pieken Numerov")
    plt.plot(E_vals[top3_born], sigma_born[top3_born], 'm^', markersize=8, zorder=5, label="Top 3 Pieken Born")
    
    # Dit plot de partiele golven
    for l in range(l_max + 1):
        plt.plot(E_vals, partial_gen[l], linestyle='--', alpha=0.7, label=f"l = {l}")
    
    # Dit plot the drie hoogste pieken voor elke methode
    # Numerove
    for idx in top3_gen:
        plt.annotate(f"{E_vals[idx]:.2f} meV", (E_vals[idx], sigma_gen[idx]),
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, fontweight='bold', color='red')
    # En Born
    for idx in top3_born:
        plt.annotate(f"{E_vals[idx]:.2f} meV", (E_vals[idx], sigma_born[idx]),
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9, color='purple')

    plt.title("Gegeneraliseerde Numerov: Verstrooiing & Faseverschuivingen", fontweight='bold')
    plt.xlabel("Energie E (meV)")
    plt.ylabel("Doorsnede (Ã…Â²)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.show()

    # Deze volgende plot is een plot voor de effectieve potentiaal
    # Het bevat ook een inzoom gedeelte, wat heel professioneel lijkt.
    r_pot = np.linspace(0.8, 4.5, 1000)
    fig, ax = plt.subplots(figsize=(10, 6))
    energy_levels = {4: 0.4639, 5: 1.4077, 6: 2.5789}
    resonance_colors = {}

    for l in range(7):
        V_eff = effective_potential(l, r_pot)
        lw = 2.5 if l in [4, 5, 6] else 1.2
        alpha_v = 1.0 if l in [4, 5, 6] else 0.6
        line = ax.plot(r_pot, V_eff, linewidth=lw, alpha=alpha_v, label=f"$l = {l}$")[0]
        if l in energy_levels:
            resonance_colors[l] = line.get_color()

    for l, E_res in energy_levels.items():
        ax.plot(r_pot, np.full_like(r_pot, E_res), color=resonance_colors[l], linestyle="--", linewidth=1.5, label=rf"$E_{{l={l}}} = {E_res:.4f}$ meV")

    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Effectieve Potentiaal $V_{\text{eff}}(r)$ met Inzoom op BarriÃ¨res")
    ax.set_xlabel("Radiale afstand $r$ ($\sigma$)")
    ax.set_ylabel("Potentiaal (meV)")
    ax.set_xlim(0.8, 4.5)
    ax.set_ylim(-6.5, 15.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, ncol=2)

    axins = ax.inset_axes([0.42, 0.5, 0.5, 0.45])
    for l in range(4, 7):
        axins.plot(r_pot, effective_potential(l, r_pot), color=resonance_colors[l], linewidth=2)
    for l, E_res in energy_levels.items():
        axins.plot(r_pot, np.full_like(r_pot, E_res), color=resonance_colors[l], linestyle="--", linewidth=1.5)
    axins.set_xlim(1.0, 2.2)
    axins.set_ylim(-1.0, 3.5)
    axins.grid(True, alpha=0.3)
    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.7)
    plt.tight_layout()
    plt.show()

# Opmerking: Ik moet nog snel even twee plots maken van de radiale golffunctie 
# bij resonantie-energieen en niet-resonantie-energieen. Ik ga dat voor de efficientie
# gewoon hieronder zetten. Coding gods, pls forgive me.

# We maken het rooster
r_wf = np.linspace(0.5, 10.0, 3000)

# De beginvoorwaarden van de radiale golffunctie
const_asymp = np.sqrt(FACTOR_2M_HBAR2 * EPSILON / 25.0)
u0_w = np.exp(-const_asymp * (r_wf[0]**(-5)))
u1_w = np.exp(-const_asymp * (r_wf[1]**(-5)))
    
# We gaan deze zes energieen vergelijken
# We hebben de drie eerder gevonden resonantie-energieen
# en dan nog drie energie die zeker geen resonantie-energie zijn.
# Ik koos deze drie, maar ik geef toe: de keuze is arbitrair.
comparisons = [
    {"l": 4, "E_res": 0.4639, "E_off": 1.2000, "label": "l = 4"},
    {"l": 5, "E_res": 1.4077, "E_off": 2.2000, "label": "l = 5"},
    {"l": 6, "E_res": 2.5789, "E_off": 3.5000, "label": "l = 6"}
]
colors_c = ['tab:blue', 'tab:orange', 'tab:green']

# Deze plot is voor de resonantie-energieen.
plt.figure(figsize=(10, 5.5))
for idx, comp in enumerate(comparisons):
    F_res = F_scattering(comp["l"], r_wf, comp["E_res"])
    u_res = generalized_numerov_prof_forward(r_wf, F_res, u0_w, u1_w)
    u_res /= np.max(np.abs(u_res))  # Normalisatie
        
    plt.plot(r_wf, u_res, color=colors_c[idx], linewidth=2.2,
             label=f"Resonantie {comp['label']} ($E = {comp['E_res']:.4f}$ meV)")
        
plt.title("Radiale golffuncties bij resonantie-energieën ($l=4, 5, 6$)", fontsize=12, fontweight='bold')
plt.xlabel("Radiale afstand $r$ ($\sigma$)", fontsize=11)
plt.ylabel("Genormaliseerde amplitude $u(r)$", fontsize=11)
plt.axhline(0, color='black', linestyle=':', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.show()

# Deze plot is voor de NIET-resonantie-energieen.
plt.figure(figsize=(10, 5.5))
for idx, comp in enumerate(comparisons):
    F_off = F_scattering(comp["l"], r_wf, comp["E_off"])
    u_off = generalized_numerov_prof_forward(r_wf, F_off, u0_w, u1_w)
    u_off /= np.max(np.abs(u_off))  # Normalisatie
        
    plt.plot(r_wf, u_off, color=colors_c[idx], linestyle='--', linewidth=2.0,
            label=f"Niet-resonant {comp['label']} ($E = {comp['E_off']:.4f}$ meV)")
        
plt.title("Radiale golffuncties bij niet-resonante energieën", fontsize=12, fontweight='bold')
plt.xlabel("Radiale afstand $r$ ($\sigma$)", fontsize=11)
plt.ylabel("Genormaliseerde amplitude $u(r)$", fontsize=11)
plt.axhline(0, color='black', linestyle=':', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.show()

########## Einde code ##############
