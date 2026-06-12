import sqlite3

def wstawianie_danych( czas, poprawnosc,id_wyniku):
    if(type(poprawnosc) != float or (type(czas) == float) or (type(id_wyniku) == int)):
        return -1
    db = sqlite3.connect('wyniki.db')
    cursor = db.cursor()
    query = F""" INSERT INTO seria (id_wyniku, dlugosc,poprawnosc) VALUES ({id_wyniku},{czas},{poprawnosc})"""
    cursor.execute(query)

def pobierz_statystyki(okres):
    if(type(okres) != str):
        return -1
    db = sqlite3.connect('wyniki.db')
    cursor = db.cursor()

    if okres == 'dzisiaj':
        query = """
            SELECT SUM(ilosc_serii), AVG(srednia_poprawnosc_serii), AVG(sredni_czas_serii) 
            FROM podsumowanie_wynikow WHERE date(data_wykonania) = date('now', 'localtime')
        """
    elif okres == 'tydzien':
        query = """
            SELECT SUM(ilosc_serii), AVG(srednia_poprawnosc_serii), AVG(sredni_czas_serii) 
            FROM podsumowanie_wynikow WHERE date(data_wykonania) >= date('now', '-7 days', 'localtime')
        """
    else:  # globalnie
        query = """
            SELECT SUM(ilosc_serii), AVG(srednia_poprawnosc_serii), AVG(sredni_czas_serii) 
            FROM podsumowanie_wynikow
        """

    cursor.execute(query)
    wynik = cursor.fetchone()
    db.close()

    return {
        "liczba_serii": wynik[0] or 0,  # "or 0" zabezpiecza przed zwróceniem None, gdy nie ma danych
        "srednia_poprawnosc": wynik[1] or 0.0,
        "sredni_czas": wynik[2] or 0.0
    }

def tworzenie_tabel():
    db = sqlite3.connect('wyniki.db')
    cursor = db.cursor()

    # 1. Tabela główna - przechowuje tylko podstawowe informacje o treningu
    cursor.execute("""
           CREATE TABLE IF NOT EXISTS wyniki (
               lp INTEGER PRIMARY KEY AUTOINCREMENT,
               data_wykonania DATETIME DEFAULT CURRENT_TIMESTAMP
           )
       """)

    # 2. Tabela szczegółowa - przechowuje dane dla każdej wykonanej serii
    cursor.execute("""
           CREATE TABLE IF NOT EXISTS seria (
               lp INTEGER PRIMARY KEY AUTOINCREMENT,            
               id_wyniku INTEGER NOT NULL,
               dlugosc FLOAT NOT NULL,
               poprawnosc FLOAT NOT NULL,
               FOREIGN KEY(id_wyniku) REFERENCES wyniki(lp)
           )
       """)
    db.commit()
    db.close()

def tworzenie_widoku():
    db = sqlite3.connect('wyniki.db')
    cursor = db.cursor()
    #Widok (VIEW) - wirtualna tabela wyliczająca statystyki w locie
    cursor.execute("""
                CREATE VIEW IF NOT EXISTS podsumowanie_wynikow AS
                SELECT 
                    w.lp AS id_treningu,
                    w.data_wykonania,
                    COUNT(s.lp) AS ilosc_serii,
                    AVG(s.poprawnosc) AS srednia_poprawnosc_serii,
                    AVG(s.dlugosc) AS sredni_czas_serii
                FROM wyniki w
                LEFT JOIN seria s ON w.lp = s.id_wyniku
                GROUP BY w.lp
            """)
    db.commit()
    db.close()
    
if __name__ == '__main__':
    tworzenie_tabel()
    tworzenie_widoku()
