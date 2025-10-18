class Logger:
    """
    Singleton Pattern - o singură instanță în toată aplicația
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        """
        Controlează crearea instanței - MAGIA Singleton!
        """
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Inițializează doar o dată, chiar dacă se apelează de mai multe ori
        """
        if not self._initialized:
            self.logs = []
            self._initialized = True




    def log(self, level, message):
        """Adaugă mesaj în log"""
        import datetime
        entry = {
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'level': level,
            'message': message
        }
        self.logs.append(entry)
        print(f"[{entry['timestamp']} {level}] {message}")

    def get_logs(self):
        """Returnează toate log-urile"""
        return self.logs

    def clear_logs(self):
        """Curăță log-urile"""
        self.logs.clear()


# Utilizare - încercăm să creăm mai multe instanțe:
logger1 = Logger()
logger2 = Logger()
logger3 = Logger()

# TOATE sunt aceeași instanță!
print(logger1 is logger2)  # True
print(logger2 is logger3)  # True
print(id(logger1) == id(logger2))  # True

# Toate scriu în același loc:
logger1.log("INFO", "Mesaj de la logger1")
logger2.log("ERROR", "Mesaj de la logger2")
logger3.log("WARNING", "Mesaj de la logger3")

# Toate au aceleași log-uri:
print(len(logger1.get_logs()))  # 3
print(len(logger2.get_logs()))  # 3
print(len(logger3.get_logs()))  # 3

# 🎯 FOLOSIT în aplicații reale pentru:
# - Configurații globale
# - Conexiuni la baza de date
# - Cache-uri globale
# - System logs