# Image de base légère
FROM python:3.10-slim

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système (nécessaire pour certains packages Python)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code
COPY . .

# Port exposé par Flask
EXPOSE 5000

# Commande de lancement (avec host 0.0.0.0 pour être accessible depuis l'extérieur)
CMD ["python", "app.py"]