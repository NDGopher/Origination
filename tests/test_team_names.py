"""Tests for team name canonicalization across Big 5 sources."""

from origination.utils.team_names import TeamNameMapper


def test_man_city_aliases():
    m = TeamNameMapper()
    assert m.canonicalize("Man City") == "Manchester City"
    assert m.canonicalize("Manchester City") == "Manchester City"
    assert m.canonicalize("Nott'm Forest") == "Nottingham Forest"
    assert m.canonicalize("Wolves") == "Wolves"


def test_bundesliga_fd_understat_aliases():
    m = TeamNameMapper()
    pairs = [
        ("Leverkusen", "Bayer Leverkusen", "Bayer Leverkusen"),
        ("Dortmund", "Borussia Dortmund", "Borussia Dortmund"),
        ("M'gladbach", "Borussia M.Gladbach", "Borussia M.Gladbach"),
        ("Ein Frankfurt", "Eintracht Frankfurt", "Eintracht Frankfurt"),
        ("FC Koln", "FC Cologne", "FC Cologne"),
        ("Heidenheim", "FC Heidenheim", "FC Heidenheim"),
        ("Mainz", "Mainz 05", "Mainz 05"),
        ("RB Leipzig", "RasenBallsport Leipzig", "RB Leipzig"),
        ("St Pauli", "St. Pauli", "St. Pauli"),
        ("Stuttgart", "VfB Stuttgart", "VfB Stuttgart"),
        ("Hamburg", "Hamburger SV", "Hamburger SV"),
        ("Hertha", "Hertha Berlin", "Hertha Berlin"),
        ("Bielefeld", "Arminia Bielefeld", "Arminia Bielefeld"),
        ("Fortuna Dusseldorf", "Fortuna Duesseldorf", "Fortuna Duesseldorf"),
        ("Greuther Furth", "Greuther Fuerth", "Greuther Fuerth"),
        ("Nurnberg", "Nuernberg", "Nuernberg"),
    ]
    for fd, us, canon in pairs:
        assert m.canonicalize(fd) == canon, fd
        assert m.canonicalize(us) == canon, us


def test_serie_a_and_la_liga_aliases():
    m = TeamNameMapper()
    assert m.canonicalize("Milan") == m.canonicalize("AC Milan") == "AC Milan"
    assert m.canonicalize("Inter") == m.canonicalize("Inter Milan") == "Inter"
    assert m.canonicalize("Verona") == m.canonicalize("Hellas Verona") == "Hellas Verona"
    assert m.canonicalize("Parma Calcio 1913") == m.canonicalize("Parma") == "Parma"
    assert m.canonicalize("Ath Bilbao") == m.canonicalize("Athletic Club") == "Athletic Club"
    assert m.canonicalize("Ath Madrid") == m.canonicalize("Atletico Madrid") == "Atletico Madrid"
    assert m.canonicalize("Betis") == m.canonicalize("Real Betis") == "Real Betis"
    assert m.canonicalize("Vallecano") == m.canonicalize("Rayo Vallecano") == "Rayo Vallecano"
    assert m.canonicalize("Espanol") == m.canonicalize("Espanyol") == "Espanyol"


def test_primeira_pinnacle_vs_fd_aliases():
    m = TeamNameMapper()
    assert m.canonicalize("Sporting CP") == m.canonicalize("Sp Lisbon") == "Sporting Lisbon"
    assert m.canonicalize("Vitoria Guimaraes") == m.canonicalize("Guimaraes") == "Vitoria Guimaraes"
    assert m.canonicalize("Estrela da Amadora") == m.canonicalize("Estrela") == "Estrela da Amadora"
    assert m.canonicalize("Sp Braga") == "Sporting Braga"
    assert m.canonicalize("PSV") == m.canonicalize("PSV Eindhoven") == "PSV Eindhoven"
    assert m.canonicalize("For Sittard") == "Fortuna Sittard"


def test_unmapped_passthrough():
    m = TeamNameMapper()
    assert m.canonicalize("Some New Club FC") == "Some New Club FC"
