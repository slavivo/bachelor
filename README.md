# Bachelor
Bakalářská práce na téma Softwarová aplikace založená na umělé inteligenci pro real time klasifikaci pohybu
Autor: Vojtěch Slavík

Tato aplikace je podporovaná na verzi Python 3.7 a výše.

# Struktura
Složka data obsahuje vytvořený dataset pohybů.
Docs obsahuje dokumentaci kódu, který jsem vytvořil.
Src obsahuje zdrojový kód aplikace.

# Jak použít
Src obsahuje textový soubor requirements.txt, který obsahuje veškeré potřebné knihovny. Pro nainstalování knihoven se dá použít pip install -r requirements.txt

Dále se dá buď spustit train_model.py, který trénuje modely nebo classify.py, který spustí klasifikaci.

train_model může dostat dva parametry. První určuje typ modelu - 's', 'u', 'k' - kde 's' je neuronová síť (NS) s učitelem, 'u' je NS bez učitele a 'k' je K-Medoids. Druhý ('t') určuje zda se provede testování nebo se model jednou natrénuje a uloží. 

classify také může dostat dva parametry. První je úplně stejný jako u train_model. V druhém můžeme specifikovat jméno .csv souboru, který chceme klasifikovat. Neboli pokud pošleme pouze jeden parametr, začne real-time klasifikace.

Příklad spuštění testování neuronové sítě s učitelem: python3 train_model s t
Příklad real-time klasifikace za použití neuronové sítě bez učitele: python3 classify.py u
