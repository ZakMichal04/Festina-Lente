import sqlite3
import time

NAZWA_BAZY = 'wyniki.db'


def _polacz():
    """Zwraca nowe polaczenie z baza."""
    return sqlite3.connect(NAZWA_BAZY)


def tworzenie_tabel():
    db = _polacz()
    cursor = db.cursor()

    # Tabela glowna 
    cursor.execute("""
           CREATE TABLE IF NOT EXISTS wyniki (
               lp INTEGER PRIMARY KEY AUTOINCREMENT,
               data_wykonania DATETIME DEFAULT CURRENT_TIMESTAMP
           )
       """)

    #  Tabela szczegolowa 
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
    db = _polacz()
    cursor = db.cursor()
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


def inicjalizuj_baze():
    tworzenie_tabel()
    tworzenie_widoku()


def utworz_trening():
    db = _polacz()
    cursor = db.cursor()
    cursor.execute("INSERT INTO wyniki DEFAULT VALUES")
    db.commit()
    nowy_id = cursor.lastrowid
    db.close()
    return nowy_id


def wstawianie_danych(czas, poprawnosc, id_wyniku):
    if not isinstance(id_wyniku, int):
        return -1
    try:
        czas = float(czas)
        poprawnosc = float(poprawnosc)
    except (TypeError, ValueError):
        return -1

    db = _polacz()
    cursor = db.cursor()
    query = "INSERT INTO seria (id_wyniku, dlugosc, poprawnosc) VALUES (?, ?, ?)"
    cursor.execute(query, (id_wyniku, czas, poprawnosc))
    db.commit()
    nowe_lp = cursor.lastrowid
    db.close()
    return nowe_lp


def pobierz_statystyki(okres):
    if type(okres) != str:
        return -1
    db = _polacz()
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
        "liczba_serii": wynik[0] or 0,  # "or 0" zabezpiecza przed None gdy brak danych
        "srednia_poprawnosc": wynik[1] or 0.0,
        "sredni_czas": wynik[2] or 0.0
    }


class RejestratorSerii:

    def __init__(self, min_czas=2.0, przerwa=1.5):
        inicjalizuj_baze()
        self.id_treningu = utworz_trening()
        self.min_czas = min_czas
        self.przerwa = przerwa
        self._aktywna = False
        self._start = 0.0
        self._ostatnia_deska = 0.0
        self._klatki = 0
        self._klatki_ok = 0

    def update(self, is_plank, bez_bledow, teraz=None):
        if teraz is None:
            teraz = time.perf_counter()

        if is_plank:
            if not self._aktywna:
                # poczatek nowej serii
                self._aktywna = True
                self._start = teraz
                self._klatki = 0
                self._klatki_ok = 0
            self._ostatnia_deska = teraz
            self._klatki += 1
            if bez_bledow:
                self._klatki_ok += 1
        else:
            # deska zniknela na wystarczajaco dlugo to zamykamy serie
            if self._aktywna and (teraz - self._ostatnia_deska) >= self.przerwa:
                self._zakoncz_serie()

    def _zakoncz_serie(self):
        if not self._aktywna:
            return
        czas = self._ostatnia_deska - self._start
        poprawnosc = (self._klatki_ok / self._klatki * 100.0) if self._klatki else 0.0
        self._aktywna = False
        if czas >= self.min_czas:
            wstawianie_danych(czas, poprawnosc, self.id_treningu)
            return czas, poprawnosc
        return None

    def zakoncz(self):
        return self._zakoncz_serie()


if __name__ == '__main__':
    inicjalizuj_baze()
    print("Baza zainicjalizowana (tabele + widok).")
