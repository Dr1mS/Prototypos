# Paper — Order Effects Without Attractors

Deux versions du meme article (identiques au contenu pres de la langue) :

- `paper.tex` — anglais
- `paper_fr.tex` — francais

## Construire

Depuis ce dossier :

```bash
# 1) figures (le script a besoin de G0.py, qui est dans ../src)
copy ..\src\G0.py .        # Windows   (ou: set PYTHONPATH=..\src)
python make_figs.py        # -> fig1..fig4      (labels anglais)
python make_figs.py --fr   # -> fr_fig1..fr_fig4 (labels francais)

# 2) papiers (deux passes pour les references croisees)
pdflatex paper.tex && pdflatex paper.tex
pdflatex paper_fr.tex && pdflatex paper_fr.tex
```

Dependances Python : `numpy`, `matplotlib`.
Dependances LaTeX : distribution standard (TeX Live / MiKTeX).
La version francaise n'utilise PAS `babel-french` (noms de sections et cesures
definis a la main) — elle compile donc sur une installation minimale.

## A COMPLETER avant diffusion

1. **Bibliographie** — le bloc `thebibliography` est un placeholder en rouge dans
   les deux fichiers. Aucune reference n'a ete inventee. Il faut retrouver et
   verifier les sources primaires pour : persona drift, tool drift, mesure de la
   derive sous memoire persistee (dependance au modele), securite de la memoire
   des agents, sensibilite au ton / politesse.
2. **Nom, affiliation, contact** — placeholders `[Author name]` / `[Nom de
   l'auteur]` en tete de chaque fichier.
3. **URL du depot** — placeholder `[repository URL]` / `[URL du depot]` dans la
   section Reproductibilite. C'est le point le plus important : l'argument du
   papier repose sur des pre-enregistrements horodates AVANT leurs donnees dans
   l'historique git de ce depot.

## Chiffres cites dans le papier — d'ou ils viennent

| chiffre | source |
|---|---|
| ecart d'appreciation -0.737 (IC95 [-0.865, -0.609]) | G1 (`results/g1_report.md`) |
| fidelite champ engineered 0.013, 10/10 | G1.5 (`results/g15_fidelity_results.json`) |
| echelle : 3.7 / 78.2 / 99.5 percentile | G3 (`results/g3_ladder.json`, `g3_null_*.json`) |
| dispersions 0.071 / 0.046 / 0.051 / 0.062 / 0.764 | G3 (`g3_ladder.json`) |
| reach-back 58.3 %, 1.36x taux de base | G3.5 (`results/g35_decision.json`, cle `provenance`) |
| retard moyen -0.114, bras miroir -0.056 | G3.5 (`g35_decision.json`) |
| ajustement R3 a = -0.76 (monostable) | G3.5 (`results/g35_fit_R3.json`) |
| fidelite champ naturel 0.933, 0/5 | G2 (`results/g2_fidelity.json`) |
| reference jamais pressee 0.688 ; bras presses 0.81-0.93 | G3.5 (`g35_decision.json`, cle `ref`) |
| niveaux de champ 0.625 / 0.792 / 0.875 / 1.000 / 1.000 | G3.5 (`results/g35_field_R3.json`) |

## Note sur la conclusion

La derniere phrase des deux versions est volontairement incisive
(« l'agent qui vous a tenu tete n'est pas celui dont il faut se mefier »).
Elle est precedee du hedge sur la replication, donc defendable — mais c'est un
choix de ton. A rendre plus sobre si le registre ne convient pas.

## Calibrage des affirmations

Ce qui peut etre affirme vs ce qui doit rester nuance est detaille dans le
squelette (`preprint_skeleton_v2.md`). Les trois pieges principaux :

- ne PAS ecrire que l'etat explicite est *requis* — l'echelle donne un faisceau
  en faveur de la necessite, jamais une preuve. Ecrire *suffisant la ou la
  persistance memoire ne l'est pas*.
- ne PAS appeler l'effet de R3 un « bassin » ou une « hysteresis » : R3 transmet
  l'ordre SANS structure d'attracteurs (dispersion dans la bande de bruit).
- ne PAS ressusciter la « reactance a la pression permissive » — retractee
  apres le balayage multi-modeles (G2.5).
