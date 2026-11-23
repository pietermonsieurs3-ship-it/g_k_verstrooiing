# Naam: Monsieurs Pieter
# E-mail: s0224946@ad.ua.ac.be
# Personen: Solo, met AI-assistentie
# Titel: README-bestand voor Gevorderde Kwantummechanica: Kwantumverstrooiing H-Kr
# Laatste bewerking: 23/11/2026 DONE PIETER

------------------------------------------------------------------------------------------------------------------

# Opmerking 1: Projectstructuur & downloaden

Om mijn oplossing te runnen, dien je beide Python-bestanden te downloaden in dezelfde map (bijvoorbeeld je "Downloads"-map):
1. `g_k_main.py` — Bevat alle numerieke kernsolvers (klassieke, matrix- en gegeneraliseerde Numerov-integratie), de Lennard-Jones-potentiaal met centrifugale barrière, faseverschuivingen en de Born-approximatie.
2. `g_k_test.py` — Het hoofduitvoeringsscript dat de simulaties draait, de energieniveaus en FWHM-/levensduurtabellen in de terminal print, en de grafieken genereert.

Bewaar beide bestanden samen in dezelfde map. Dit is nodig, aangezien `g_k_test.py` de functies direct importeert vanuit `g_k_main.py`.

------------------------------------------------------------------------------------------------------------------

# Opmerking 2: Omgeving & uitvoeren

Je kunt de code uitvoeren vanuit je terminal. Het wordt aanbevolen om een schone virtuele omgeving aan te maken om bibliotheekconflicten te voorkomen. Volg deze instructies (ervan uitgaande dat beide bestanden in je "Downloads"-map staan):

cd ~/Downloads                                 # Navigeer naar de map met beide scripts
python3 -m venv quantum_env                    # Maak de virtuele omgeving aan
source quantum_env/bin/activate                # Activeer de omgeving (Linux/macOS)
pip install numpy scipy matplotlib             # Installeer de vereiste wetenschappelijke bibliotheken
python3 g_k_test.py                            # Voer het hoofdscript uit om alle taken te verrichten

(Opmerking voor Windows-gebruikers: activeer de omgeving met `qm_scattering_env\Scripts\activate` in plaats van `source ...`).

------------------------------------------------------------------------------------------------------------------

# Opmerking 3: Uitvoering

Tijdens de uitvoering is geen handmatige toetsenbordinput vereist; het script voert alle berekeningen sequentieel uit.

------------------------------------------------------------------------------------------------------------------

# Opmerking 4: Tip voor de terminalweergave

Omdat het script gedetailleerde tabellen met meerdere kolommen print (zoals de energie- en FWHM-vergelijkingstabellen), wordt het aanbevolen om je terminalvenster iets breder te maken of het lettertype van de terminal licht te verkleinen (bijv. via het menu rechtsboven of met `Ctrl` + `-`), zodat de kolommen niet overspringen naar nieuwe regels.

------------------------------------------------------------------------------------------------------------------

# Opmerking 5: PEP-8

Wegens tijdsbeperking voldoet de code grotendeels niet aan de PEP-8 standaard style conventie.
Dit heeft geen invloed op de functionaliteit van de code, maar enkel op het uiterlijk van de code zelf.

------------------------------------------------------------------------------------------------------------------
