Dépendances
===========
* python3
* pip3

À exécuter de préférence dans un venv :
```
pip3 install -r requirements.txt
```

CartPole
--------
Pour tester dans l'environnement CartPole-V1 :
```
python3 CartPole-v1.py
```

Breakeout Atari
---------------
La liste des paramètres est accessible avec la commande suivante :
```
python3 BreakoutNoFrameskip-v4.py --help
```

| Paramètre           | Description                              |
|---------------------|------------------------------------------|
|   `--cuda`          |   active cuda                            |
| `--autosave n`      | sauvegarde le modèle tous les n épisodes |
| `-s` / `--save nom` | sauvegarde le modèle nom.pt              |
| `-l n` / `--learn n`| lance n épisode d'apprentissage          |
| `-t n` / `--test n` | lance n épisode de test                  |

Par exemple pour lancer 500 épisodes d'apprentissage, 200 de tests avec cuda dans le fichier model.pt :
```
python3 BreakoutNoFrameskip-v4.py --cuda --learn 500 -t 200 -s model
```
On peut par exemple reprendre l'apprentissage en faisant :
```
python3 BreakoutNoFrameskip-v4.py model.pt --cuda --learn 500
```
