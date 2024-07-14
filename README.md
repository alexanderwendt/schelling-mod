# schelling-mod
Modified version of Schelling's Segregation simulator that includes economic factors.

# Credits
This is a modified version of the Schelling's simualtor version with Streamlit from here: https://github.com/adilmoujahid/streamlit-schelling. 
Thanks to ADIL MOUJAHID. Blog: https://adilmoujahid.com/.

# Setup
Install Python 3.11 with the provided ```requirements.txt```

# TODOs
1. Similarity-Threshold ist eine Verteilung und nicht ein einzelner Wert. Tau wert einstellbar für jede Gruppe
2. Kulturelle Unterschiedlichkeit ist unterschiedlich für unterschiedliche Gruppen. Gruppe G1-G2: 1,2; G1-G3: 1,5; G1-G4: 2,5
Matrix für jede Gruppe mit Ähnlichkeiten zum nachschauen. World value survey als Quelle verwenden für Ähnlichkeiten.
3. Ökonomie: jede Gruppe hat eine Einkommensverteilung. Das ist ein Ähnlichkeitsattribut. Gleiche Gruppen suchen einander, also wie ein Ähnlichkeitsattribut.
4. Lagewert: Der Wert des Hauses ist 50%*20 Jahreseinkommen der unmittelbaren Nachbarn, die existieren.
5. Lagezuschlagsfaktor: jedes Haus hat ein Grundwert, basierend auf Lage von Faktor 0,5 bis 1,5 und dieser Wert ist fix für jede Zelle. Der Wert wird mit dem Einkommenspreis multipliziert.
6. Einkommenszufriedenheit: wenn Wohnkosten also Kosten des Hauses > 60%*20 Jahreseinkommen ist, kann man sich das Haus nicht leisten und wird unzufrieden und muss umziehen.
7. Umzug: wenn jemand unzufrieden ist, kann diese Person nur in ein leeres Haus, dessen Wert <= 60%*20 Jahreseinkommen kostet.


# Literature
To read: https://www.sciencedirect.com/science/article/pii/S0264275124000520
