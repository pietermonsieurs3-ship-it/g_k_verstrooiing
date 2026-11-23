"""Houd deze infrastructuur main code in de Downloads folder."""

# Naam: Monsieurs Pieter
# E-mail: s0224946@ad.ua.ac.be
# Solo, met AI-assistentie
# Titel: Main code voor gevorderde kwantum taak DONE PIETER

# Opmerking: Ik heb nog andere dingen geprobeerd, maar wees gerust, deze code omvat alle zaken die in het verslag 
# tevoorschijn komen, alle data, alle plots. Hetgeen dat niet relevant was of niet gelukt, tja, dat
# is verwijderd van de final versie, en maken daarom geen deel uit van mijn indiening.

# Importeer deze bibliotheken
import numpy as np  # Om roosterpunten aan te maken
import scipy.linalg as la  # Om schrodinger op te lossen (Matrix-Numerov)
import scipy.special as sp  # Voor sferische Bessel functies
import scipy.signal as signal  # Om resonantiepieken te vinden
from scipy.integrate import quad  # Voor numerieke integratie (Born)

# Kies deze constanten, zoals vermeld op opgaveblad BB.
EPSILON = 5.9  # meV
SIGMA = 1.0  # Eenheid van lengte = 1 sigma (3.75 Å)
FACTOR_2M_HBAR2 = 6.12 # meV^-1 * sigma^-2
CONV_ANG_SQ = 3.75**2  # Omrekenfactor naar Å^2

def lennard_jones_potential(r):
    """Bereken de Lennard-Jones potentiaal."""
    # Gebruik de definitie, zoals vermeld op het opgaveblad BB.
    r_safe = np.maximum(r, 1e-8)
    term = SIGMA / r_safe
    # Geef nu het resultaat terug, we gaan het nodig hebben
    return EPSILON * (term**12 - 2.0 * term**6)

def effective_potential(l, r):
    """Bereken de totale effectieve potentiaal: V_eff(r) = V_LJ(r) + V_centrifugaal."""
    V_lj = lennard_jones_potential(r)
    V_cent = (l * (l + 1.0)) / (FACTOR_2M_HBAR2 * r**2)
    # Geef nu het resultaat terug, we gaan het nodig hebben
    return V_lj + V_cent

def F_scattering(l, r, E):
    """Berekent de functie F(r), voor de gegeneraliseerde Numerov-methode"""
    V_lj = lennard_jones_potential(r)
    V_cent = (l * (l + 1.0)) / (FACTOR_2M_HBAR2 * r**2)
    return FACTOR_2M_HBAR2 * (V_lj + V_cent - E)

# De volgende methoden zijn de solvers van de vergelijkingen

# Opmerking: Ik had origineel de volgende methoden opgesteld:
# - Klassiek Numerove (met een heel lange breuk formule)
# - Matrix Numerov
# - Generaliseerd Numerov (met de gegeven formule)

# Ik verstond pas later dat de eerste en derde methode eigenlijk op hetzelfde 
# neerkomen, waardoor ze equivalent zijn. Mijn fout. We gaan daarom enkel door met 
# Generaliseerd Numerove en beide matrix-numerovs methodes.

def classical_numerov_forward(x, k2, y0, y1):
    """Verwezenlijk de klassieke Numerov voorwaartse integratie."""
    n = len(x)
    h = x[1] - x[0]
    h2_12 = (h**2) / 12.0
    y = np.zeros(n)
    y[0], y[1] = y0, y1
    for i in range(1, n - 1):
        # Deze supergrote breuk formule
        factor_curr = 2.0 * (1.0 - 5.0 * h2_12 * k2[i]) * y[i]
        factor_prev = (1.0 + h2_12 * k2[i-1]) * y[i-1]
        factor_next = 1.0 + h2_12 * k2[i+1]
        y[i+1] = (factor_curr - factor_prev) / factor_next
    return y

def matrix_numerov(x, V_mat):
    """Verwezenlijk de Matrix-Numerov methode voor sferische potentialen (eigenwaardeprobleem)."""
    # Dit is in sferische coordinaten trouwens.
    
    n = len(x)
    h = x[1] - x[0]
    num_internal = n - 1
    
    # Eerst matrix A klaarmaken
    A = np.zeros((num_internal, num_internal))
    np.fill_diagonal(A, -2.0)
    np.fill_diagonal(A[1:], 1.0)
    np.fill_diagonal(A[:, 1:], 1.0)
    A = A / (h**2)
    
    # Dan matrix B klaarmaken
    B = np.zeros((num_internal, num_internal))
    np.fill_diagonal(B, 10.0 / 12.0)
    np.fill_diagonal(B[1:], 1.0 / 12.0)
    np.fill_diagonal(B[:, 1:], 1.0 / 12.0)
    
    # Nu Hamiltoniaan maken, met A en B.
    H = -0.5 * (la.inv(B) @ A) + np.diag(V_mat[:-1])
    H = 0.5 * (H + H.T)
    
    # Los de Hamiltoniaan op.
    energies, wavefunctions_internal = la.eigh(H)
    wavefunctions = np.zeros((n, num_internal))
    wavefunctions[:-1, :] = wavefunctions_internal

    return energies, wavefunctions

def matrix_numerov_cartesian_1d(x_grid, V_1d):
    """Voer de Matrix-Numerov methode uit voor 1D Cartesische roosters."""
    # Dit is in cartesische coordinaten nu.

    n = len(x_grid)
    h = x_grid[1] - x_grid[0]
    num_internal = n - 2
    
    # Eerste matrix A opstellen
    A = np.zeros((num_internal, num_internal))
    np.fill_diagonal(A, -2.0)
    np.fill_diagonal(A[1:], 1.0)
    np.fill_diagonal(A[:, 1:], 1.0)
    A = A / (h**2)
    
    # Dan matrix B klaarmaken
    B = np.zeros((num_internal, num_internal))
    np.fill_diagonal(B, 10.0 / 12.0)
    np.fill_diagonal(B[1:], 1.0 / 12.0)
    np.fill_diagonal(B[:, 1:], 1.0 / 12.0)
    
    # Hamiltoniaan maken, met A en B
    H = -0.5 * (la.inv(B) @ A) + np.diag(V_1d[1:-1])
    H = 0.5 * (H + H.T)
    
    # We lossen de hamiltoniaan nu op.
    energies_1d, wf_internal = la.eigh(H)
    wf_1d = np.zeros((n, num_internal))
    wf_1d[1:-1, :] = wf_internal
    
    return energies_1d, wf_1d

def generalized_numerov_prof_forward(r, F_array, u0, u1):
    """Doe de degeneraliseerde Numerov methode (voorwaarts) volgens de prof."""
    n = len(r)
    h = r[1] - r[0]
    u = np.zeros(n)
    w = np.zeros(n)
    
    # Formules komen van opgaveblad BB.
    u[0], u[1] = u0, u1
    w[0] = (1.0 - (h**2 / 12.0) * F_array[0]) * u[0]
    w[1] = (1.0 - (h**2 / 12.0) * F_array[1]) * u[1]
    
    for i in range(1, n - 1):
        # Formules komen van opgaveblad BB, waarvan voorwaarts.
        w[i+1] = 2.0 * w[i] - w[i-1] + (h**2) * F_array[i] * u[i]
        u[i+1] = w[i+1] / (1.0 - (h**2 / 12.0) * F_array[i+1])
    return u

def generalized_numerov_prof_backward(r, F_array, u_end, u_next):
    """Doe de gegeneraliseerde Numerov methode (achterwaarts) volgens de prof."""
    # Deze code is principieel hetzelfde, maar nu achterwaarts, dus met minus.
    
    n = len(r)
    h = r[1] - r[0]
    u = np.zeros(n)
    w = np.zeros(n)

    # Formules komen van opgaveblad BB.
    u[-1], u[-2] = u_end, u_next
    w[-1] = (1.0 - (h**2 / 12.0) * F_array[-1]) * u[-1]
    w[-2] = (1.0 - (h**2 / 12.0) * F_array[-2]) * u[-2]
    
    for i in range(n - 2, 0, -1):
        # Formules komen van opgaveblad BB, waarvan achterwaarts.
        w[i-1] = 2.0 * w[i] - w[i+1] + (h**2) * F_array[i] * u[i]
        u[i-1] = w[i-1] / (1.0 - (h**2 / 12.0) * F_array[i-1])
    return u

def calculate_phase_shift(l, E, r_grid, u_sol):
    """Bereken de faseverschuiving delta_l uit de asymptotische golffunctie."""
    
    # Bereken eerst het golwiggetal k
    k = np.sqrt(FACTOR_2M_HBAR2 * E)
    
    # Index getallen, om specifieke posities te selecteren
    idx1, idx2 = -50, -1
    
    # De radiale afstanden en golffunctiewaarden
    r1, r2 = r_grid[idx1], r_grid[idx2]
    u1, u2 = u_sol[idx1], u_sol[idx2]
    
    # Verhoudingsfactor K
    # Louter om de berekening later wat leesbaarder te maken.
    K = (r1 * u2) / (r2 * u1)
    
    # Dit definieert de sferische Bessel-functies
    jl_1 = sp.spherical_jn(l, k * r1)
    jl_2 = sp.spherical_jn(l, k * r2)
    
    # Dit definieert de sferische Neumann-functies
    nl_1 = sp.spherical_yn(l, k * r1)
    nl_2 = sp.spherical_yn(l, k * r2)
    
    # The tangens van de faseverschuiving wordt gegeven door deze fractie
    tan_delta = (K * jl_1 - jl_2) / (K * nl_1 - nl_2)
    
    # De hoek wordt daaruit gehaald via arctan().
    return np.arctan(tan_delta)

def compute_cross_section_method_with_printing(method_type, E_range, l_max=6, r_min=0.5, r_max=15.0, n_points=3000):
    """Bereken de totale werkzame doorsnede sigma tot en print faseverschuivingen op scherm."""
    
    # Rooster definieren
    r_grid = np.linspace(r_min, r_max, n_points)
    
    # Randvoorwaarden opstellen
    const_asymp = np.sqrt(FACTOR_2M_HBAR2 * EPSILON / 25.0)
    u0 = np.exp(-const_asymp * (r_grid[0]**(-5)))
    u1 = np.exp(-const_asymp * (r_grid[1]**(-5)))
    
    # Lege lijst aanmaken, want we gaan hem straks vullen
    sigma_tot_list = []
    partial_sigma = np.zeros((l_max + 1, len(E_range)))
    
    # Mooie tabel om te zien wat er precies geprint wordt.
    print(f"\n==================================================")
    print(f" RESULTATEN FASEVERSCHUIVINGEN: {method_type.upper()} NUMEROV")
    print(f"==================================================")
    print(f"{'E (meV)':<10} | {'l':<5} | {'Delta_l (rad)':<15} | {'Delta_l (deg)':<15}")
    print("-" * 55)
    
    # Dit loopt over alle energieen
    for idx_E, E in enumerate(E_range):
        k2_val = FACTOR_2M_HBAR2 * E
        sum_cross_section = 0.0
        
        # Dit loopt over alle draai-impulsmomenten
        # Terwijl we nog in de loop van de energieen zijn.
        for l in range(l_max + 1):
        
            # F(r) eerste definieren
            F_arr = F_scattering(l, r_grid, E)
            
            # Als de methode in kwestie de generaliseerd methode is,
            # dan gaan we die functie aanroepen en uitvoeren.
            # Indien niet, dan gaat de klassieke numerov uitgevoerd worden.
            # Voor het verstrooiingsprobleem, gaan we enkel door met de voorwaartste versie.
            # omdat o.a. de nul-randvoorwaarde ontbreekt, dus langs achter gaan heeft geen zin.
            if method_type == 'generalized':
                u_sol = generalized_numerov_prof_forward(r_grid, F_arr, u0, u1)
            else:
                k2_arr = -F_arr
                u_sol = classical_numerov_forward(r_grid, k2_arr, u0, u1)
            
            # We berekenen nu de faseverschuiving.
            delta_l = calculate_phase_shift(l, E, r_grid, u_sol)
            
            # Elke 100ste energie, printen we het resultaat.
            if idx_E % 100 == 0:
                print(f"{E:<10.3f} | {l:<5} | {delta_l:<15.5f} | {np.degrees(delta_l):<15.2f}")
            
            # Dit is de werkzame doorsnede voor deze partiele golf
            sigma_l = (4.0 * np.pi / k2_val) * (2.0 * l + 1.0) * (np.sin(delta_l)**2)
            partial_sigma[l, idx_E] = sigma_l * CONV_ANG_SQ
            sum_cross_section += sigma_l
        
        # Wanneer over alle l's geloopt is, gaan we het resultaat toevoegen aan 
        # de lege lijst van sigma_tot.
        sigma_tot_list.append(sum_cross_section * CONV_ANG_SQ)
        
    return r_grid, E_range, np.array(sigma_tot_list), partial_sigma

def born_scattering_cross_section_multiwav(E_range, l_max=6):
    """Bereken de doorsnede via de Eerste Born-benadering voor alle partiële golven."""
    
    # We gaan deze lege lijst nog vullen
    sigma_born_tot_list = []
    
    # We lopen over alle energieen
    for E in E_range:
        k = np.sqrt(FACTOR_2M_HBAR2 * E)
        k2_val = FACTOR_2M_HBAR2 * E
        sum_sigma_born = 0.0
        
        # We loopen over alle draai-impulsmomenten
        for l in range(l_max + 1):
        
            # Het integrandum binnen de eerste Born-benadering moeten we
            # even snel nog definieren. Daarvoor hebben we de Lennard-Jones 
            # potentiaal en Bessel-functie nodig.
            def integrand(r):
                """"Maak even het integrandum voor Born."""
                V = lennard_jones_potential(r)
                jl = sp.spherical_jn(l, k * r)
                return V * (jl**2) * (r**2)
            
            # Dit is numerieke integratie toegepast
            integral, _ = quad(integrand, 0.5, 15.0, limit=100)
            
            # Dit geeft de uiteindelijke faseverschuiving, met Born
            delta_born_l = -FACTOR_2M_HBAR2 * k * integral
            
            # Dit is de werkzame doorsnede voor deze faseverschuiving
            sigma_l_born = (4.0 * np.pi / k2_val) * (2.0 * l + 1.0) * (np.sin(delta_born_l)**2)
            sum_sigma_born += sigma_l_born
        
        # Als we gedaan zijn met alle l's, dan vullen we het resultaat aan
        # aan de lijst van sigma_tot.
        sigma_born_tot_list.append(sum_sigma_born * CONV_ANG_SQ)
        
    return np.array(sigma_born_tot_list)

def compute_born_phase_shifts_with_printing(target_energies, l_max=6):
    # Docstring voor wat uitleg van wat deze methode doet.
    """Bereken en print de faseverschuivingen voor Born delta_l."""
    
    # Deze code is net zoals hiervoor maar nu met Born.
    l_vals = np.arange(0, l_max + 1)
    born_delta_results = {}
    
    # Even een mooie tabel maken, met Borns naam.
    print(f"\n==================================================")
    print(f" RESULTATEN FASEVERSCHUIVINGEN: BORN-BENADERING")
    print(f"==================================================")
    print(f"{'E (meV)':<10} | {'l':<5} | {'Delta_l (rad)':<15} | {'Delta_l (deg)':<15}")
    print("-" * 55)
    
    # loop over de energieen
    for E in target_energies:
        k = np.sqrt(FACTOR_2M_HBAR2 * E)
        delta_list = []
        
        # Loop over de draai-impulsmomenten
        for l in l_vals:
            
            # Definieer snel het integrandum
            def integrand(r):
                """"Maak even het integrandum voor Born."""
                V = lennard_jones_potential(r)
                jl = sp.spherical_jn(l, k * r)
                return V * (jl**2) * (r**2)
            
            # Numerieke integratie toegepast
            integral, _ = quad(integrand, 0.5, 20.0, limit=100)
            
            # Fase verschuiving berekend
            delta_born_l = -FACTOR_2M_HBAR2 * k * integral
            
            # Misschien interessant, omgezet naar graden
            delta_deg = np.degrees(delta_born_l)
            
            # Dit toevoegen aan de delta_l lijst.
            delta_list.append(delta_deg)
            
            # En dan printen we het.
            print(f"{E:<10.3f} | {l:<5} | {delta_born_l:<15.5f} | {delta_deg:<15.2f}")
            
        born_delta_results[E] = np.array(delta_list)
        
    return l_vals, born_delta_results

########## Einde code ##############
