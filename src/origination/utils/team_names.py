"""
Central team-name canonicalization.

All sources (football-data, Understat, FBref) map into a single canonical name.
Silent mismatches destroy join quality — fail loud when unmapped.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from loguru import logger

# Canonical name -> aliases across sources (football-data + Understat + common variants).
# Both FD abbreviations and Understat titles must appear as aliases of the same canonical.
_TEAM_ALIASES: dict[str, list[str]] = {
    # --- EPL / English ---
    "Arsenal": ["Arsenal", "Arsenal FC"],
    "Aston Villa": ["Aston Villa", "Aston Villa FC"],
    "Bournemouth": ["Bournemouth", "AFC Bournemouth"],
    "Brentford": ["Brentford", "Brentford FC"],
    "Brighton": ["Brighton", "Brighton & Hove Albion", "Brighton and Hove Albion"],
    "Burnley": ["Burnley", "Burnley FC"],
    "Cardiff": ["Cardiff", "Cardiff City"],
    "Chelsea": ["Chelsea", "Chelsea FC"],
    "Coventry": ["Coventry", "Coventry City", "Coventry City FC"],
    "Crystal Palace": ["Crystal Palace", "Crystal Palace FC"],
    "Everton": ["Everton", "Everton FC"],
    "Fulham": ["Fulham", "Fulham FC"],
    "Huddersfield": ["Huddersfield", "Huddersfield Town"],
    "Hull": ["Hull", "Hull City"],
    "Ipswich": ["Ipswich", "Ipswich Town"],
    "Leeds": ["Leeds", "Leeds United"],
    "Leicester": ["Leicester", "Leicester City"],
    "Liverpool": ["Liverpool", "Liverpool FC"],
    "Luton": ["Luton", "Luton Town"],
    "Manchester City": ["Man City", "Manchester City", "Manchester City FC"],
    "Manchester United": ["Man United", "Manchester United", "Manchester Utd", "Man Utd"],
    "Middlesbrough": ["Middlesbrough", "Middlesbrough FC"],
    "Newcastle": ["Newcastle", "Newcastle United", "Newcastle Utd"],
    "Norwich": ["Norwich", "Norwich City"],
    "Nottingham Forest": ["Nott'm Forest", "Nottingham Forest", "Forest"],
    "QPR": ["QPR", "Queens Park Rangers"],
    "Sheffield United": ["Sheffield United", "Sheffield Utd"],
    "Sheffield Wednesday": ["Sheffield Weds", "Sheffield Wednesday"],
    "Southampton": ["Southampton", "Southampton FC"],
    "Stoke": ["Stoke", "Stoke City"],
    "Sunderland": ["Sunderland", "Sunderland AFC"],
    "Swansea": ["Swansea", "Swansea City"],
    "Tottenham": ["Tottenham", "Tottenham Hotspur", "Spurs"],
    "Watford": ["Watford", "Watford FC"],
    "West Brom": ["West Brom", "West Bromwich Albion", "West Bromwich"],
    "West Ham": ["West Ham", "West Ham United"],
    "Wolves": ["Wolves", "Wolverhampton Wanderers", "Wolverhampton"],
    # --- Bundesliga (D1) ---
    "Augsburg": ["Augsburg", "FC Augsburg"],
    "Bayer Leverkusen": ["Leverkusen", "Bayer Leverkusen", "Bayer 04 Leverkusen"],
    "Bayern Munich": ["Bayern Munich", "Bayern München", "Bayern"],
    "Arminia Bielefeld": ["Bielefeld", "Arminia Bielefeld"],
    "Bochum": ["Bochum", "VfL Bochum"],
    "Borussia Dortmund": ["Dortmund", "Borussia Dortmund"],
    "Borussia M.Gladbach": [
        "M'gladbach",
        "Mgladbach",
        "Borussia M.Gladbach",
        "Borussia Monchengladbach",
        "Borussia Mönchengladbach",
        "Gladbach",
    ],
    "Darmstadt": ["Darmstadt", "Darmstadt 98", "SV Darmstadt 98"],
    "Eintracht Frankfurt": ["Ein Frankfurt", "Eintracht Frankfurt", "Frankfurt"],
    "Elversberg": ["Elversberg", "SV Elversberg"],
    "FC Cologne": ["FC Koln", "FC Cologne", "Cologne", "Köln", "FC Köln", "1. FC Koln", "1. FC Köln"],
    "FC Heidenheim": ["Heidenheim", "FC Heidenheim", "1. FC Heidenheim"],
    "Fortuna Duesseldorf": [
        "Fortuna Dusseldorf",
        "Fortuna Duesseldorf",
        "Fortuna Düsseldorf",
        "Dusseldorf",
    ],
    "Freiburg": ["Freiburg", "SC Freiburg"],
    "Greuther Fuerth": ["Greuther Furth", "Greuther Fuerth", "Greuther Fürth", "Furth"],
    "Hamburger SV": ["Hamburg", "Hamburger SV", "HSV"],
    "Hannover 96": ["Hannover", "Hannover 96", "Hanover"],
    "Hertha Berlin": ["Hertha", "Hertha Berlin", "Hertha BSC"],
    "Hoffenheim": ["Hoffenheim", "TSG Hoffenheim"],
    "Holstein Kiel": ["Holstein Kiel", "Kiel"],
    "Ingolstadt": ["Ingolstadt", "FC Ingolstadt", "Ingolstadt 04"],
    "Mainz 05": ["Mainz", "Mainz 05", "FSV Mainz 05", "1. FSV Mainz 05"],
    "Nuernberg": ["Nurnberg", "Nuernberg", "Nürnberg", "1. FC Nurnberg", "1. FC Nürnberg"],
    "Paderborn": ["Paderborn", "SC Paderborn", "Paderborn 07"],
    "RB Leipzig": ["RB Leipzig", "RasenBallsport Leipzig", "Rasenballsport Leipzig", "Leipzig"],
    "Schalke 04": ["Schalke 04", "Schalke", "FC Schalke 04"],
    "St. Pauli": ["St Pauli", "St. Pauli", "FC St. Pauli", "FC St Pauli"],
    "VfB Stuttgart": ["Stuttgart", "VfB Stuttgart"],
    "Union Berlin": ["Union Berlin", "1. FC Union Berlin", "FC Union Berlin"],
    "Werder Bremen": ["Werder Bremen", "Bremen"],
    "Wolfsburg": ["Wolfsburg", "VfL Wolfsburg"],
    # --- Serie A (I1) ---
    "AC Milan": ["Milan", "AC Milan", "A.C. Milan"],
    "Atalanta": ["Atalanta", "Atalanta Bergamasca"],
    "Bologna": ["Bologna", "Bologna FC"],
    "Cagliari": ["Cagliari", "Cagliari Calcio"],
    "Como": ["Como", "Como 1907"],
    "Cremonese": ["Cremonese", "US Cremonese"],
    "Empoli": ["Empoli", "Empoli FC"],
    "Fiorentina": ["Fiorentina", "ACF Fiorentina"],
    "Frosinone": ["Frosinone", "Frosinone Calcio"],
    "Genoa": ["Genoa", "Genoa CFC"],
    "Hellas Verona": ["Verona", "Hellas Verona", "Hellas Verona FC"],
    "Inter": ["Inter", "Inter Milan", "Internazionale", "FC Internazionale"],
    "Juventus": ["Juventus", "Juventus FC"],
    "Lazio": ["Lazio", "SS Lazio"],
    "Lecce": ["Lecce", "US Lecce"],
    "Monza": ["Monza", "AC Monza"],
    "Napoli": ["Napoli", "SSC Napoli"],
    "Parma": ["Parma", "Parma Calcio", "Parma FC", "Parma Calcio 1913"],
    "Pisa": ["Pisa", "AC Pisa", "Pisa SC", "ASC Pisa"],
    "Roma": ["Roma", "AS Roma", "A.S. Roma"],
    "Salernitana": ["Salernitana", "US Salernitana"],
    "Sampdoria": ["Sampdoria", "UC Sampdoria"],
    "Sassuolo": ["Sassuolo", "US Sassuolo"],
    "Spezia": ["Spezia", "Spezia Calcio"],
    "Torino": ["Torino", "Torino FC"],
    "Udinese": ["Udinese", "Udinese Calcio"],
    "Venezia": ["Venezia", "Venezia FC"],
    "Benevento": ["Benevento", "Benevento Calcio"],
    "Brescia": ["Brescia", "Brescia Calcio"],
    "Crotone": ["Crotone", "FC Crotone"],
    "Chievo": ["Chievo", "Chievo Verona"],
    "SPAL": ["SPAL", "SPAL 2013", "Spal"],
    "Pescara": ["Pescara", "Delfino Pescara"],
    "Carpi": ["Carpi", "Carpi FC"],
    "Palermo": ["Palermo", "US Palermo", "Palermo FC"],
    "Cesena": ["Cesena", "Cesena FC"],
    "Catania": ["Catania", "Calcio Catania"],
    # --- La Liga (SP1) ---
    "Alaves": ["Alaves", "Alavés", "Deportivo Alaves", "Deportivo Alavés"],
    "Almeria": ["Almeria", "Almería", "UD Almeria", "UD Almería"],
    "Athletic Club": ["Ath Bilbao", "Athletic Club", "Athletic Bilbao", "Athletic"],
    "Atletico Madrid": [
        "Ath Madrid",
        "Atletico Madrid",
        "Atlético Madrid",
        "Atletico de Madrid",
        "Atlético de Madrid",
    ],
    "Barcelona": ["Barcelona", "FC Barcelona", "Barca"],
    "Cadiz": ["Cadiz", "Cádiz", "Cadiz CF", "Cádiz CF"],
    "Celta Vigo": ["Celta", "Celta Vigo", "Celta de Vigo", "RC Celta"],
    "Eibar": ["Eibar", "SD Eibar"],
    "Elche": ["Elche", "Elche CF"],
    "Espanyol": ["Espanol", "Espanyol", "RCD Espanyol"],
    "Getafe": ["Getafe", "Getafe CF"],
    "Girona": ["Girona", "Girona FC"],
    "Granada": ["Granada", "Granada CF"],
    "Las Palmas": ["Las Palmas", "UD Las Palmas"],
    "Leganes": ["Leganes", "Leganés", "CD Leganes", "CD Leganés"],
    "Levante": ["Levante", "Levante UD"],
    "Mallorca": ["Mallorca", "RCD Mallorca"],
    "Osasuna": ["Osasuna", "CA Osasuna"],
    "Rayo Vallecano": ["Vallecano", "Rayo Vallecano", "Rayo"],
    "Real Betis": ["Betis", "Real Betis", "Betis Sevilla"],
    "Real Madrid": ["Real Madrid", "R. Madrid"],
    "Real Sociedad": ["Sociedad", "Real Sociedad"],
    "Real Valladolid": ["Valladolid", "Real Valladolid"],
    "Sevilla": ["Sevilla", "Sevilla FC"],
    "Valencia": ["Valencia", "Valencia CF"],
    "Villarreal": ["Villarreal", "Villarreal CF"],
    "Deportivo La Coruna": [
        "La Coruna",
        "Deportivo La Coruna",
        "Deportivo La Coruña",
        "Deportivo",
        "Depor",
    ],
    "Malaga": ["Malaga", "Málaga", "Malaga CF", "Málaga CF"],
    "Sporting Gijon": ["Sp Gijon", "Sporting Gijon", "Sporting Gijón", "Sporting de Gijon"],
    "Huesca": ["Huesca", "SD Huesca"],
    "Cordoba": ["Cordoba", "Córdoba", "Cordoba CF"],
    "Oviedo": ["Oviedo", "Real Oviedo"],
    "Racing Santander": [
        "Racing Santander",
        "Santander",
        "Racing de Santander",
        "Real Racing Club",
        "Racing",
    ],
    # --- Primeira Liga (P1) — FD abbreviations vs Pinnacle / fixture titles ---
    "Sporting Lisbon": [
        "Sp Lisbon",
        "Sporting CP",
        "Sporting Lisbon",
        "Sporting Clube de Portugal",
    ],
    "Sporting Braga": ["Sp Braga", "Braga", "Sporting Braga", "SC Braga"],
    "Vitoria Guimaraes": [
        "Guimaraes",
        "Vitoria Guimaraes",
        "Vitória Guimarães",
        "Vitoria SC",
    ],
    "Estrela da Amadora": ["Estrela", "Estrela da Amadora", "Estrela Amadora", "CF Estrela"],
    "Alverca": ["Alverca", "FC Alverca"],
    "Benfica": ["Benfica", "SL Benfica"],
    "Porto": ["Porto", "FC Porto"],
    "Casa Pia": ["Casa Pia", "Casa Pia AC"],
    "Gil Vicente": ["Gil Vicente", "Gil Vicente FC"],
    "Moreirense": ["Moreirense", "Moreirense FC"],
    "Rio Ave": ["Rio Ave", "Rio Ave FC"],
    "Santa Clara": ["Santa Clara", "CD Santa Clara"],
    "Nacional": ["Nacional", "CD Nacional"],
    "Arouca": ["Arouca", "FC Arouca"],
    "Famalicao": ["Famalicao", "Famalicão", "FC Famalicao"],
    "Tondela": ["Tondela", "CD Tondela"],
    "Estoril": ["Estoril", "Estoril Praia", "GD Estoril"],
    "AVS": ["AVS", "AVS Futebol", "AVS SAD"],
    # --- Eredivisie ---
    "PSV Eindhoven": ["PSV", "PSV Eindhoven", "PSV Eindhoven FC"],
    "Fortuna Sittard": ["For Sittard", "Fortuna Sittard"],
    "Utrecht": ["Utrecht", "FC Utrecht"],
    "NEC Nijmegen": ["Nijmegen", "NEC Nijmegen", "NEC"],
    "PEC Zwolle": ["Zwolle", "PEC Zwolle"],
    "ADO Den Haag": ["Den Haag", "ADO Den Haag", "ADO"],
    "Cambuur": ["Cambuur", "Cambuur Leeuwarden", "Cambuur Leeuwaarden", "SC Cambuur"],
    "AZ Alkmaar": ["AZ Alkmaar", "AZ"],
    "Sparta Rotterdam": ["Sparta Rotterdam", "Sparta"],
    "Go Ahead Eagles": ["Go Ahead Eagles", "Go Ahead"],
    # --- Ligue 1 ---
    "Paris Saint-Germain": ["Paris Saint-Germain", "Paris SG", "PSG", "Paris Saint Germain"],
    "Marseille": ["Marseille", "Olympique Marseille", "OM"],
    "Lyon": ["Lyon", "Olympique Lyonnais", "OL"],
    "Monaco": ["Monaco", "AS Monaco"],
    "Lille": ["Lille", "LOSC Lille", "LOSC"],
    "Nice": ["Nice", "OGC Nice"],
    "Rennes": ["Rennes", "Stade Rennais", "Stade Rennes"],
    "Lens": ["Lens", "RC Lens"],
    "Strasbourg": ["Strasbourg", "RC Strasbourg", "Racing Strasbourg"],
    "Nantes": ["Nantes", "FC Nantes"],
    "Toulouse": ["Toulouse", "Toulouse FC"],
    "Brest": ["Brest", "Stade Brestois", "Stade Brestois 29"],
    "Reims": ["Reims", "Stade de Reims"],
    "Montpellier": ["Montpellier", "Montpellier HSC"],
    "Lorient": ["Lorient", "FC Lorient"],
    "Angers": ["Angers", "Angers SCO"],
    "Auxerre": ["Auxerre", "AJ Auxerre"],
    "Le Havre": ["Le Havre", "Havre", "Le Havre AC"],
    "Troyes": ["Troyes", "ES Troyes", "Troyes AC"],
    "Paris FC": ["Paris FC", "PFC"],
    "Le Mans": ["Le Mans", "Le Mans FC"],
    # --- Belgium ---
    "Club Brugge": ["Club Brugge", "Club Bruges", "Club Brugge KV"],
    "Cercle Brugge": ["Cercle Brugge", "Cercle"],
    "Anderlecht": ["Anderlecht", "RSC Anderlecht"],
    "Genk": ["Genk", "KRC Genk", "Racing Genk"],
    "Standard Liege": ["Standard Liege", "Standard Liège", "Standard"],
    "Royal Antwerp": ["Royal Antwerp", "Antwerp", "Antwerp FC"],
    "Gent": ["Gent", "KAA Gent", "La Gantoise"],
    "Charleroi": ["Charleroi", "Sporting Charleroi"],
    "Mechelen": ["Mechelen", "KV Mechelen", "Yellow-Red Mechelen"],
    "Zulte Waregem": ["Zulte Waregem", "Zulte-Waregem", "SV Zulte Waregem"],
    "Beveren": ["Beveren", "SK Beveren", "Waasland-Beveren"],
    "Westerlo": ["Westerlo", "KVC Westerlo"],
    "RAAL La Louviere": ["RAAL La Louviere", "RAAL La Louvière", "La Louviere", "La Louvière"],
    "Lommel": ["Lommel", "Lommel SK", "Lommel United"],
    "Union Saint-Gilloise": ["Union Saint-Gilloise", "Union SG", "Royale Union SG"],
    # --- MLS ---
    "Atlanta United": ["Atlanta United", "Atlanta Utd", "Atlanta"],
    "Austin FC": ["Austin FC", "Austin"],
    "CF Montreal": ["CF Montreal", "Montreal", "Montreal Impact", "Club de Foot Montreal"],
    "Charlotte FC": ["Charlotte FC", "Charlotte"],
    "Chicago Fire": ["Chicago Fire", "Chicago"],
    "FC Cincinnati": ["FC Cincinnati", "Cincinnati"],
    "Colorado Rapids": ["Colorado Rapids", "Colorado"],
    "Columbus Crew": ["Columbus Crew", "Columbus"],
    "FC Dallas": ["FC Dallas", "Dallas"],
    "D.C. United": ["D.C. United", "DC United", "D.C United"],
    "Houston Dynamo": ["Houston Dynamo", "Houston"],
    "Inter Miami": ["Inter Miami", "Inter Miami CF", "Miami"],
    "LA Galaxy": ["Los Angeles Galaxy", "LA Galaxy", "Galaxy", "Los Angeles Galaxy"],
    "Los Angeles FC": ["Los Angeles FC", "LAFC", "Los Angeles FC"],
    "Minnesota United": ["Minnesota United", "Minnesota Utd", "Minnesota"],
    "Nashville SC": ["Nashville SC", "Nashville"],
    "New England Revolution": ["New England Revolution", "New England", "NE Revolution"],
    "New York City": ["New York City", "NYCFC", "New York City FC"],
    "New York Red Bulls": ["New York Red Bulls", "NY Red Bulls", "New York RB"],
    "Orlando City": ["Orlando City", "Orlando"],
    "Philadelphia Union": ["Philadelphia Union", "Philadelphia"],
    "Portland Timbers": ["Portland Timbers", "Portland"],
    "Real Salt Lake": ["Real Salt Lake", "Salt Lake", "RSL"],
    "San Diego FC": ["San Diego FC", "San Diego"],
    "San Jose Earthquakes": ["San Jose Earthquakes", "San Jose", "SJ Earthquakes"],
    "Seattle Sounders": ["Seattle Sounders", "Seattle", "Seattle Sounders FC"],
    "Sporting Kansas City": ["Sporting Kansas City", "Sporting KC", "Kansas City"],
    "St Louis City": ["St Louis City SC", "St. Louis City", "St Louis City", "Saint Louis City"],
    "Toronto FC": ["Toronto FC", "Toronto"],
    "Vancouver Whitecaps": ["Vancouver Whitecaps", "Vancouver", "Whitecaps"],
    # --- Championship / EFL extras ---
    "Blackburn": ["Blackburn", "Blackburn Rovers"],
    "Bolton": ["Bolton", "Bolton Wanderers"],
    "Preston": ["Preston", "Preston North End"],
    "Lincoln": ["Lincoln", "Lincoln City"],
    "Wrexham": ["Wrexham", "Wrexham AFC"],
    "Birmingham": ["Birmingham", "Birmingham City"],
    "Portsmouth": ["Portsmouth", "Portsmouth FC"],
    "Bristol City": ["Bristol City", "Bristol C"],
    "Millwall": ["Millwall", "Millwall FC"],
    "Charlton": ["Charlton", "Charlton Athletic"],
    "Derby": ["Derby", "Derby County"],
}


# Build reverse map (lowercase alias -> canonical)
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canon, aliases in _TEAM_ALIASES.items():
    _ALIAS_TO_CANONICAL[canon.casefold()] = canon
    for a in aliases:
        _ALIAS_TO_CANONICAL[a.casefold()] = canon


class TeamNameMapper:
    """Map arbitrary source team names to canonical names."""

    def __init__(self, extra: dict[str, str] | None = None) -> None:
        self._map = dict(_ALIAS_TO_CANONICAL)
        if extra:
            for alias, canon in extra.items():
                self._map[alias.casefold()] = canon

    def canonicalize(self, name: str, *, strict: bool = False) -> str:
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return name
        key = str(name).strip().casefold()
        if key in self._map:
            return self._map[key]
        if strict:
            raise KeyError(f"Unmapped team name: {name!r}")
        logger.warning("Unmapped team name (passthrough): {!r}", name)
        return str(name).strip()

    def map_series(self, s: pd.Series, *, strict: bool = False) -> pd.Series:
        return s.map(lambda x: self.canonicalize(x, strict=strict))

    def unmapped(self, names: Iterable[str]) -> list[str]:
        out: list[str] = []
        for n in names:
            if n is None or (isinstance(n, float) and pd.isna(n)):
                continue
            if str(n).strip().casefold() not in self._map:
                out.append(str(n))
        return sorted(set(out))


DEFAULT_MAPPER = TeamNameMapper()
