import os

# Создаём каталоги на диске, если их нет
os.makedirs("/var/data/puzzles", exist_ok=True)
os.makedirs("/var/data/states", exist_ok=True)
