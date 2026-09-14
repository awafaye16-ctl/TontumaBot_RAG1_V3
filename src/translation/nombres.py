"""Conversion des nombres français écrits en lettres vers des chiffres.

Pourquoi : NLLB traduit mal les nombres en toutes lettres vers le wolof.
Mesuré sur bilalfaye/nllb-200-distilled-600M-wo-fr-en :

    « Les frais s'élèvent à cinq mille francs. »  ->  « Junni ... »        (1 000)
    « Les frais s'élèvent à 5000 francs. »        ->  « ... 5000 francs »  (5 000)

Une division par cinq sur un montant que l'usager va payer au guichet. Le
`SYSTEM_PROMPT_PLAIN` demande déjà au LLM d'écrire en chiffres ; ce module est
le filet en dessous, pour les fois où il ne le fait pas — et pour les documents
ingérés, qui ne passent par aucun prompt.

Portée volontairement limitée aux cardinaux : ce sont les montants, délais et
quantités. Les ordinaux (« premier », « deuxième ») sont laissés tels quels,
ils se traduisent correctement et ne portent aucun risque financier.
"""
import re

from journal import journal

_log = journal("nombres")

_UNITES = {
    "zéro": 0, "zero": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4,
    "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10,
    "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15,
    "seize": 16, "vingt": 20, "vingts": 20, "trente": 30, "quarante": 40,
    "cinquante": 50, "soixante": 60,
}
_CENT  = {"cent": 100, "cents": 100}
_MILLE = {"mille": 1000, "milles": 1000}
_GRANDS = {"million": 10**6, "millions": 10**6,
           "milliard": 10**9, "milliards": 10**9}

_MOTS_NOMBRE = set(_UNITES) | set(_CENT) | set(_MILLE) | set(_GRANDS)
_LIAISON     = {"et"}

# Un « un » isolé est presque toujours l'article, pas la quantité : le convertir
# transformerait « Un montant de... » en « 1 montant de... ».
_ARTICLES = {"un", "une"}

_JETON   = re.compile(r"[A-Za-zÀ-ÿ]+")
_ADJACENT = re.compile(r"^[-\s]+$")      # ce qui peut séparer deux mots d'un nombre


def _valeur(mots: list[str]) -> int | None:
    """Valeur d'une suite de mots-nombres, ou None si la suite est incohérente."""
    total = courant = 0
    precedent = None
    vu = False

    for mot in mots:
        if mot in _LIAISON:
            continue
        vu = True
        if mot in _UNITES:
            v = _UNITES[mot]
            # « quatre-vingt(s) » : le vingt multiplie le quatre qui précède
            if v == 20 and precedent == 4:
                courant = courant - 4 + 80
            else:
                courant += v
            precedent = v
        elif mot in _CENT:
            courant = (courant or 1) * 100
            precedent = 100
        elif mot in _MILLE:
            total += (courant or 1) * 1000
            courant, precedent = 0, 1000
        elif mot in _GRANDS:
            total = (total + (courant or 1)) * _GRANDS[mot]
            courant, precedent = 0, _GRANDS[mot]
        else:
            return None

    return (total + courant) if vu else None


def en_chiffres(texte: str) -> str:
    """Remplace chaque nombre écrit en lettres par son écriture chiffrée."""
    if not texte:
        return texte

    jetons = list(_JETON.finditer(texte))
    if not jetons:
        return texte

    morceaux, position, i = [], 0, 0
    while i < len(jetons):
        if jetons[i].group().lower() not in _MOTS_NOMBRE:
            i += 1
            continue

        # Étendre tant que les mots suivants appartiennent au nombre ET que
        # rien d'autre qu'un tiret ou une espace ne les sépare.
        j = i + 1
        while j < len(jetons):
            suivant = jetons[j].group().lower()
            entre   = texte[jetons[j - 1].end():jetons[j].start()]
            if not _ADJACENT.match(entre) or suivant not in (_MOTS_NOMBRE | _LIAISON):
                break
            j += 1
        # Une liaison « et » ne peut pas terminer un nombre
        while j > i and jetons[j - 1].group().lower() in _LIAISON:
            j -= 1

        mots = [t.group().lower() for t in jetons[i:j]]
        valeur = _valeur(mots)
        if valeur is None or (len(mots) == 1 and mots[0] in _ARTICLES):
            i = j if j > i else i + 1
            continue

        morceaux.append(texte[position:jetons[i].start()])
        morceaux.append(str(valeur))
        position = jetons[j - 1].end()
        i = j

    morceaux.append(texte[position:])
    return "".join(morceaux)


# =============================================================================
#  Sens inverse : chiffres -> nombres wolof, pour le TTS
# =============================================================================
#  Mesuré par aller-retour (Oolel-Voices puis re-transcription HuBERT-CTC) :
#  le TTS ne sait PAS lire les chiffres — « 15 » ressort en « fiscal », « 5000 »
#  en « se am sene ». Écrits en toutes lettres wolof, les mêmes nombres
#  reviennent intacts. Les chiffres doivent donc être rendus au wolof AVANT la
#  synthèse, alors qu'ils doivent rester en chiffres pour NLLB.
#
#  Les tables ci-dessous ont été relues et validées par un locuteur wolof
#  (relecture du 10 septembre 2026) : formes de liaison en -i, dizaines
#  composées, et la convention monétaire en dërëm plus bas. Elles restent
#  isolées de la logique pour rester corrigibles si un usage local diffère.

_WO_UNITES = {
    0: "tus", 1: "benn", 2: "ñaar", 3: "ñett", 4: "ñeent", 5: "juróom",
    6: "juróom-benn", 7: "juróom-ñaar", 8: "juróom-ñett", 9: "juróom-ñeent",
}
# Forme de liaison (devant un nom compté ou un multiplicateur) : ñaar -> ñaari
_WO_LIAISON = {
    1: "benn", 2: "ñaari", 3: "ñetti", 4: "ñeenti", 5: "juróomi",
    6: "juróom-benni", 7: "juróom-ñaari", 8: "juróom-ñetti", 9: "juróom-ñeenti",
}


def _wo_sous_cent(n: int) -> str:
    """1 à 99 en wolof. Base décimale sur un socle quinaire (juróom = 5)."""
    if n < 10:
        return _WO_UNITES[n]
    dizaines, reste = divmod(n, 10)
    base = "fukk" if dizaines == 1 else f"{_WO_LIAISON[dizaines]} fukk"
    return base if reste == 0 else f"{base} ak {_WO_UNITES[reste]}"


def _wo_sous_mille(n: int) -> str:
    """1 à 999 en wolof."""
    if n < 100:
        return _wo_sous_cent(n)
    centaines, reste = divmod(n, 100)
    base = "téeméer" if centaines == 1 else f"{_WO_LIAISON[centaines]} téeméer"
    return base if reste == 0 else f"{base} ak {_wo_sous_cent(reste)}"


def _wo_nombre(n: int) -> str:
    """Écriture wolof d'un entier positif (jusqu'au million exclu)."""
    if n == 0:
        return _WO_UNITES[0]
    if n < 1000:
        return _wo_sous_mille(n)
    milliers, reste = divmod(n, 1000)
    if milliers < 10:
        base = "junni" if milliers == 1 else f"{_WO_LIAISON[milliers]} junni"
    else:
        base = f"{_wo_sous_mille(milliers)} junni"
    return base if reste == 0 else f"{base} ak {_wo_sous_mille(reste)}"


# =============================================================================
#  Montants : le wolof compte l'argent en dërëm (1 dërëm = 5 francs)
# =============================================================================
#  Un montant ne se dit PAS comme une quantité ordinaire — le nombre prononcé
#  est le montant divisé par 5 :
#
#      5000 F  ->  junni           (littéralement « mille » : 1000 x 5 = 5000)
#       500 F  ->  téeméer         (littéralement « cent »  :  100 x 5 =  500)
#      1000 F  ->  ñaari téeméer                            (  200 x 5 = 1000)
#
#  Annoncer « juróomi junni » pour 5000 F reviendrait à dire 25 000 F.
#  Cette règle ne vaut que pour l'argent : jours, documents et photos se
#  comptent normalement (15 jours = fukk ak juróom).

_DEREM = 5

# Ce qui, dans la sortie wolof de NLLB, marque un montant. On accepte les
# variantes françaises comme wolofisées : NLLB produit les deux.
_MONNAIE = re.compile(
    # Le « cfa » optionnel porte sa propre espace : sinon un `\s*` avant un
    # groupe absent avale l'espace du mot suivant (« ... dërëmla »).
    r"\s*(?:francs?(?:\s*cfa)?|fcfa|cfa|xaalisu\s+seefaa|seefaa|dërëm)\b",
    re.IGNORECASE,
)

# Unité prononcée après le nombre d'un montant.
_UNITE_MONTANT = "dërëm"

# Un nombre, avec ou sans séparateur de milliers. Le groupement n'est reconnu
# que par tranches de EXACTEMENT trois chiffres : sans cette contrainte,
# « Am na 1 2 3 fan » se lisait comme le nombre 123.
_CHIFFRES = re.compile(r"\d{1,3}(?:[  .]\d{3})+|\d+")


def _entier(brut: str) -> int | None:
    net = re.sub(r"[\s.]", "", brut)
    return int(net) if net.isdigit() else None


def en_lettres_wolof(texte: str) -> str:
    """Remplace les nombres chiffrés par leur écriture wolof, pour le TTS.

    À n'appliquer que juste avant la synthèse : le texte affiché garde les
    chiffres, plus lisibles à l'écran et fidèles au document source.

    Un nombre suivi d'une unité monétaire est converti en dërëm (division par
    5) ; tout autre nombre est rendu en numération ordinaire.
    """
    if not texte:
        return texte

    resultat, position = [], 0
    for m in _CHIFFRES.finditer(texte):
        if m.start() < position:
            continue
        valeur = _entier(m.group())
        # Au-delà du million, l'écriture en toutes lettres devient illisible :
        # on laisse le chiffre plutôt que de produire une phrase interminable.
        if valeur is None or valeur >= 10**6:
            continue

        monnaie = _MONNAIE.match(texte, m.end())
        fin     = m.end()
        if monnaie is not None and valeur % _DEREM == 0:
            mots = f"{_wo_nombre(valeur // _DEREM)} {_UNITE_MONTANT}"
            fin  = monnaie.end()
        else:
            # Hors montant, ou montant non divisible par 5 (hors système
            # dërëm) : numération ordinaire. Mieux vaut un nombre inhabituel
            # à l'oreille qu'un montant faux au guichet.
            mots = _wo_nombre(valeur)

        resultat.append(texte[position:m.start()])
        resultat.append(mots)
        position = fin

    resultat.append(texte[position:])
    return "".join(resultat)


if __name__ == "__main__":
    CAS = [
        ("Les frais s'élèvent à cinq mille francs.",     "5000"),
        ("Le délai est de quinze jours.",                "15"),
        ("Présentez deux photos d'identité.",            "2"),
        ("Le coût est de mille francs CFA.",             "1000"),
        ("Comptez vingt et un jours.",                   "21"),
        ("Il faut quatre-vingt-dix jours.",              "90"),
        ("Quatre-vingts francs.",                        "80"),
        ("Soixante-dix jours.",                          "70"),
        ("Un montant de trois cent cinquante francs.",   "350"),
        ("Deux millions de francs.",                     "2000000"),
        ("Deux photos et quinze jours.",                 "2"),
        ("Une pièce d'identité valide.",                 None),
        ("Rien à convertir ici.",                        None),
    ]
    print("── FR en lettres -> chiffres (avant NLLB) ──────────────────────")
    ok = 0
    for source, attendu in CAS:
        sortie = en_chiffres(source)
        bon = (sortie == source) if attendu is None else (attendu in sortie)
        ok += bon
        print(f"  {'ok ' if bon else 'KO '} {source}\n      -> {sortie}")
    print(f"  {ok}/{len(CAS)} cas corrects\n")
    # Montants : le nombre prononcé est le montant divisé par 5 (dërëm).
    # Tout le reste se compte normalement.
    CAS_WO = [
        ("Njëg li 5000 francs la.",          "junni dërëm"),
        ("Njëg li 500 francs CFA la.",       "téeméer dërëm"),
        ("Njëgu timbre bi 1000 francs la.",  "ñaari téeméer dërëm"),
        ("Am na 100 francs.",                "ñaari fukk dërëm"),
        ("Fey na 25000 FCFA.",               "juróomi junni dërëm"),
        # Comptage ordinaire : aucune division
        ("Jamono 15 fan la.",                "fukk ak juróom fan"),
        ("Indil 2 nataal.",                  "ñaar nataal"),
        ("Jamono 21 fan la.",                "ñaari fukk ak benn fan"),
        ("Etaas 3 bi.",                      "ñett bi"),
    ]
    print("── chiffres -> wolof parlé (avant TTS) ─────────────────────────")
    ok_wo = 0
    for source, attendu in CAS_WO:
        sortie = en_lettres_wolof(source)
        bon = attendu in sortie
        ok_wo += bon
        print(f"  {'ok ' if bon else 'KO '} {source}\n      -> {sortie}")
    print(f"  {ok_wo}/{len(CAS_WO)} cas corrects")
# =============================================================================
#  Garde-fou : les nombres ont-ils survécu à la traduction ?
# =============================================================================
#  NLLB réécrit parfois les montants de son propre chef, et se trompe :
#  « 1000 francs CFA » ressort en « junniy dërëm » (= 5000 F). Quand le modèle
#  fabrique un montant, l'information d'origine a disparu et aucun
#  post-traitement ne peut la réparer.
#
#  Ce contrôle ne corrige donc rien : il rend l'écart VISIBLE au lieu de le
#  laisser silencieux. Chaque nombre du français est cherché dans le wolof sous
#  ses formes légitimes. Celui qu'on n'y retrouve sous aucune n'est pas
#  forcément faux, mais il n'est plus vérifiable : c'est ce que signale
#  `suspects`.
#
#  Deux pièges, tous deux rencontrés en test :
#
#  - la recherche doit se faire à la FRONTIÈRE DE MOT. « junni » se retrouve
#    à l'intérieur de « junniy », ce qui validait à tort « 1000 francs » rendu
#    par « junniy dërëm » (= 5000 F) — exactement l'erreur à détecter.
#  - pour un MONTANT, la numération ordinaire ne vaut pas acquittement : dit en
#    dërëm, un montant de 1000 F doit se lire « ñaari téeméer dërëm ». Accepter
#    « junni » (= 1000 en comptage ordinaire) laisserait passer un facteur 5.

# Liaison des multiplicateurs devant un nom compté, comme pour les unités.
_LIAISON_MOT = {"fukk": "fukki", "téeméer": "téeméeri", "junni": "junniy"}
_LIAISON_MOT.update({_WO_UNITES[v]: _WO_LIAISON[v] for v in _WO_LIAISON})

# Tout mot pouvant appartenir à un nombre wolof. Sert à rejeter un appariement
# partiel : « junni dërëm » trouvé dans « juróomi junni dërëm » désigne 25 000,
# pas 5 000 — le mot qui précède trahit un nombre plus grand.
_MOTS_WOLOF = (set(_WO_UNITES.values()) | set(_WO_LIAISON.values())
               | set(_LIAISON_MOT.values()) | {"fukk", "téeméer", "junni", "ak"})

_MOT = re.compile(r"[\w\-]+", re.UNICODE)


def _mots(texte: str) -> list[str]:
    return _MOT.findall((texte or "").lower())


def _variantes(n: int) -> set[str]:
    """Écritures acceptables d'un nombre : forme libre et forme de liaison.

    Devant un nom compté, le dernier mot prend un -i (ou -y) : « ñaar » devient
    « ñaari nataal », « fukk ak juróom » devient « fukk ak juróomi fan ».
    """
    base   = _wo_nombre(n)
    formes = {base}
    mots   = base.split()
    if mots and mots[-1] in _LIAISON_MOT:
        formes.add(" ".join(mots[:-1] + [_LIAISON_MOT[mots[-1]]]))
    return formes


def _sequence(mots_cible: list[str], forme: list[str],
              suivant: str | None = None, interdit: str | None = None) -> bool:
    """La suite de mots `forme` apparaît-elle telle quelle dans la cible ?

    `suivant` impose le mot qui doit la suivre, `interdit` celui qui ne doit
    pas. Un appariement précédé d'un autre mot-nombre est rejeté : il n'est
    alors que la fin d'un nombre plus grand.
    """
    n = len(forme)
    for i in range(len(mots_cible) - n + 1):
        if mots_cible[i:i + n] != forme:
            continue
        if i > 0 and mots_cible[i - 1] in _MOTS_WOLOF:
            continue
        apres = mots_cible[i + n] if i + n < len(mots_cible) else None
        if suivant is not None and apres != suivant:
            continue
        if interdit is not None and apres == interdit:
            continue
        return True
    return False


def _nombres_contexte(texte: str) -> list[tuple[int, bool]]:
    """[(valeur, est_un_montant)] pour chaque nombre chiffré du texte."""
    trouves = []
    for m in _CHIFFRES.finditer(texte):
        valeur = _entier(m.group())
        if valeur is None:
            continue
        trouves.append((valeur, _MONNAIE.match(texte, m.end()) is not None))
    return trouves


def verifier_nombres(source_fr: str, cible_wo: str) -> dict:
    """Compare les nombres du français source et ceux du wolof produit.

    Retourne un dict destiné à la trace :
        controles  nombre de valeurs examinées
        suspects   [{valeur, montant}] introuvables dans la cible
        ajoutes    valeurs présentes dans la cible mais absentes de la source
        coherent   True si ni suspect ni ajout
    """
    attendus = _nombres_contexte(source_fr)
    cible    = cible_wo or ""
    suspects = []

    mots_cible = _mots(cible)

    for valeur, montant in attendus:
        # Chiffres conservés tels quels : toujours acquittant. On compare les
        # VALEURS et non les graphies, « 00 » devant acquitter la valeur 0.
        acquitte = any(m.isdigit() and int(m) == valeur for m in mots_cible)

        if not acquitte and montant and valeur % _DEREM == 0:
            # Dit en dërëm : le nombre prononcé est valeur / 5, suivi de l'unité.
            acquitte = any(
                _sequence(mots_cible, f.split(), suivant=_UNITE_MONTANT)
                for f in _variantes(valeur // _DEREM)
            )
        if not acquitte and not montant:
            acquitte = any(_sequence(mots_cible, f.split())
                           for f in _variantes(valeur))
        if not acquitte and montant:
            # Montant dit en francs : numération ordinaire, mais alors PAS suivi
            # de « dërëm », sinon il est faux d'un facteur 5.
            acquitte = any(
                _sequence(mots_cible, f.split(), interdit=_UNITE_MONTANT)
                for f in _variantes(valeur)
            )

        if not acquitte:
            suspects.append({"valeur": valeur, "montant": montant})

    # Chiffres apparus dans la cible sans exister dans la source
    sources_vues = {v for v, _ in attendus}
    ajoutes = [v for v, _ in _nombres_contexte(cible) if v not in sources_vues]

    return {
        "controles": len(attendus),
        "suspects":  suspects,
        "ajoutes":   ajoutes,
        "coherent":  not suspects and not ajoutes,
    }


# =============================================================================
#  Réinjection : remettre dans le wolof les nombres du français
# =============================================================================
#  Mesuré sur les 20 questions de test : 8 réponses sur 20 portaient un nombre
#  altéré. Le plus grave, « composez le 3333 » rendu par « 333 » — le numéro
#  d'urgence interne de l'hôpital, prononcé faux par la borne.
#
#  Contrairement à ce que je croyais, ces nombres SONT récupérables : le
#  français fait autorité et les nombres y apparaissent dans le même ordre que
#  dans le wolof. On réaligne donc position par position.
#
#  La réparation n'a lieu QUE si les deux textes portent autant de nombres l'un
#  que l'autre. Si NLLB en a ajouté ou supprimé, l'alignement n'a plus de sens
#  et l'on préfère ne rien toucher : la valeur reste signalée par
#  `verifier_nombres`, mais jamais remplacée au hasard.

def realigner(source_fr: str, cible_wo: str) -> tuple[str, dict]:
    """Remplace les nombres du wolof par ceux du français, dans l'ordre.

    Retourne le texte corrigé et un bilan :
        aligne      True si les deux textes portaient autant de nombres
        corriges    [{avant, apres}] valeurs effectivement remplacées
        n_source    nombres trouvés dans le français
        n_cible     nombres trouvés dans le wolof
    """
    if not source_fr or not cible_wo:
        return cible_wo, {"aligne": False, "corriges": [],
                          "n_source": 0, "n_cible": 0}

    attendus = [v for v, _ in _nombres_contexte(source_fr)]
    presents = list(_CHIFFRES.finditer(cible_wo))

    bilan = {"aligne": len(attendus) == len(presents),
             "corriges": [], "n_source": len(attendus),
             "n_cible": len(presents), "concordance": None}
    if not bilan["aligne"] or not attendus:
        return cible_wo, bilan

    # Compter n'est pas aligner. Mesuré sur une réponse réelle portant huit
    # nombres de part et d'autre : aucun ne coïncidait, la réécriture les a
    # tous décalés d'une position et démembré un numéro de téléphone — et le
    # contrôle, voyant toutes les valeurs présentes, déclarait « cohérent ».
    #
    # On exige donc que l'alignement soit CORROBORÉ : la majorité des nombres
    # doit déjà coïncider position par position, la valeur discordante étant
    # alors une altération isolée. Un texte à un seul nombre fait exception —
    # il n'y a qu'une place, l'affectation est sans ambiguïté.
    accords = sum(1 for attendu, trouve in zip(attendus, presents)
                  if _entier(trouve.group()) == attendu)
    bilan["concordance"] = round(accords / len(attendus), 2)
    if len(attendus) > 1 and accords * 2 < len(attendus):
        bilan["aligne"] = False
        return cible_wo, bilan

    morceaux, position = [], 0
    for attendu, trouve in zip(attendus, presents):
        actuel = _entier(trouve.group())
        morceaux.append(cible_wo[position:trouve.start()])
        if actuel == attendu:
            morceaux.append(trouve.group())      # inchangé, format préservé
        else:
            morceaux.append(str(attendu))
            bilan["corriges"].append({"avant": trouve.group(), "apres": str(attendu)})
        position = trouve.end()
    morceaux.append(cible_wo[position:])

    return "".join(morceaux), bilan
