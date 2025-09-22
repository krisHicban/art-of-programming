import sys
import random
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QDialog, QTextEdit, QFrame,
                            QScrollArea, QGridLayout, QSplitter, QDesktopWidget)
from PyQt5.QtGui import QFont, QPalette
from PyQt5.QtCore import Qt, QTimer

class AIExplanationDialog(QDialog):
    def __init__(self, move_type, board_state, chosen_move, parent=None):
        super().__init__(parent)
        self.move_type = move_type
        self.board_state = board_state[:]
        self.chosen_move = chosen_move
        self.initUI()

    def initUI(self):
        self.setWindowTitle('🧠 Cum "gândește" computerul - AI simplu')
        self.setGeometry(300, 200, 700, 600)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #f8f9fa, stop: 1 #e9ecef);
            }
            QLabel {
                color: #2c3e50;
                font-family: Arial;
            }
            .title {
                font-size: 20px;
                font-weight: bold;
                color: #e74c3c;
                margin: 10px;
            }
            .step {
                font-size: 14px;
                background-color: #fff;
                border-left: 4px solid #3498db;
                padding: 12px;
                margin: 8px;
                border-radius: 4px;
            }
            .step-win {
                border-left-color: #27ae60;
                background-color: #d5f4e6;
            }
            .step-block {
                border-left-color: #f39c12;
                background-color: #fef9e7;
            }
            .step-random {
                border-left-color: #9b59b6;
                background-color: #f4ecf7;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

        layout = QVBoxLayout()

        # Title
        title = QLabel('🤖 Explicația Mutării Computerului')
        title.setObjectName('title')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            color: #ffffff;
            background-color: #2c3e50;
            padding: 15px;
            border-radius: 8px;
            margin: 15px;
        """)
        layout.addWidget(title)

        # Create scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()


        # Step by step analysis - skil for as too much crowding the UI
        # self.add_step_analysis(scroll_layout)

        # Current move explanation
        move_frame = QFrame()
        move_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {'#d5f4e6' if self.move_type == 'win' else '#fef9e7' if self.move_type == 'block' else '#f4ecf7'};
                border: 2px solid {'#27ae60' if self.move_type == 'win' else '#f39c12' if self.move_type == 'block' else '#9b59b6'};
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
            }}
        """)
        move_layout = QVBoxLayout()

        move_icons = {'win': '🎯', 'block': '🛡️', 'random': '🎲'}
        move_titles = {
            'win': 'CÂȘTIG!',
            'block': 'BLOCHEZ!',
            'random': 'MUTARE ALEATOARE!'
        }
        move_descriptions = {
            'win': 'Am găsit o mutare câștigătoare!',
            'block': 'Trebuie să te blochez!',
            'random': 'Nu e nicio urgență, aleg random.'
        }

        current_move_title = QLabel(f'{move_icons[self.move_type]} {move_titles[self.move_type]}')
        current_move_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        current_move_title.setAlignment(Qt.AlignCenter)
        move_layout.addWidget(current_move_title)

        current_move_desc = QLabel(move_descriptions[self.move_type])
        current_move_desc.setStyleSheet("font-size: 14px; font-style: italic;")
        current_move_desc.setAlignment(Qt.AlignCenter)
        move_layout.addWidget(current_move_desc)

        position_label = QLabel(f'Poziția aleasă: {self.chosen_move + 1} (rând {self.chosen_move // 3 + 1}, coloana {self.chosen_move % 3 + 1})')
        position_label.setStyleSheet("font-size: 14px; margin-top: 5px;")
        position_label.setAlignment(Qt.AlignCenter)
        move_layout.addWidget(position_label)

        move_frame.setLayout(move_layout)
        scroll_layout.addWidget(move_frame)

        # Philosophy section
        philosophy_frame = QFrame()
        philosophy_frame.setStyleSheet("""
            QFrame {
                background-color: #ecf0f1;
                border: 2px solid #95a5a6;
                border-radius: 8px;
                padding: 15px;
                margin: 10px;
            }
        """)
        philosophy_layout = QVBoxLayout()

        philo_title = QLabel('🎭 Filosofia AI: Simplu vs Complex')
        philo_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        philosophy_layout.addWidget(philo_title)

        philosophy_text = """
🔹 Acest AI este FOARTE SIMPLU - doar 3 reguli fixe!
🔹 Nu "învață" nimic, nu "înțelege" jocul
🔹 Nu calculează viitorul, nu folosește experiența trecută
🔹 E ca un robot care execută instrucțiuni simple

🧠 AI ADEVĂRAT (Rețele Neurale):
🔹 Ar putea "învăța" jocând mii de partide
🔹 Ar putea calcula 10+ mutări în viitor
🔹 Ar putea găsi strategii pe care nici nu le știm!
🔹 Dar... nu am putea înțelege exact cum "gândește"

💡 Paradoxul: Cu cât AI devine mai puternic, cu atât devine mai misterios!
        """

        philo_label = QLabel(philosophy_text)
        philo_label.setStyleSheet("font-size: 12px; line-height: 1.4;")
        philo_label.setWordWrap(True)
        philosophy_layout.addWidget(philo_label)

        philosophy_frame.setLayout(philosophy_layout)
        scroll_layout.addWidget(philosophy_frame)

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Close button
        close_btn = QPushButton('Înțeleg! 🧠')
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def add_step_analysis(self, layout):
        analysis_frame = QFrame()
        analysis_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
        """)
        analysis_layout = QVBoxLayout()

        analysis_title = QLabel('🔍 Analiza pas cu pas:')
        analysis_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        analysis_layout.addWidget(analysis_title)

        # Step 1: Check for winning move
        step1_label = QLabel('PASUL 1: Verific dacă pot câștiga...')
        step1_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #27ae60;")
        analysis_layout.addWidget(step1_label)

        can_win, win_pos = self.check_winning_move('O')
        if can_win:
            step1_result = QLabel(f'✅ DA! Pot câștiga la poziția {win_pos + 1}!')
            step1_result.setStyleSheet("background-color: #d5f4e6; padding: 8px; border-radius: 4px; margin: 5px;")
        else:
            step1_result = QLabel('❌ Nu pot câștiga acum.')
            step1_result.setStyleSheet("background-color: #fadbd8; padding: 8px; border-radius: 4px; margin: 5px;")
        analysis_layout.addWidget(step1_result)

        if not can_win:
            # Step 2: Check for blocking move
            step2_label = QLabel('PASUL 2: Verific dacă trebuie să blochez...')
            step2_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f39c12; margin-top: 10px;")
            analysis_layout.addWidget(step2_label)

            can_block, block_pos = self.check_winning_move('X')
            if can_block:
                step2_result = QLabel(f'⚠️ DA! Trebuie să blochez la poziția {block_pos + 1}!')
                step2_result.setStyleSheet("background-color: #fef9e7; padding: 8px; border-radius: 4px; margin: 5px;")
            else:
                step2_result = QLabel('✅ Nu trebuie să blochez nimic.')
                step2_result.setStyleSheet("background-color: #d5f4e6; padding: 8px; border-radius: 4px; margin: 5px;")
            analysis_layout.addWidget(step2_result)

            if not can_block:
                # Step 3: Random move
                step3_label = QLabel('PASUL 3: Aleg o poziție liberă la întâmplare...')
                step3_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #9b59b6; margin-top: 10px;")
                analysis_layout.addWidget(step3_label)

                empty_positions = [i for i, val in enumerate(self.board_state) if val == '']
                step3_result = QLabel(f'🎲 Poziții libere: {[p+1 for p in empty_positions]}')
                step3_result.setStyleSheet("background-color: #f4ecf7; padding: 8px; border-radius: 4px; margin: 5px;")
                analysis_layout.addWidget(step3_result)

        analysis_frame.setLayout(analysis_layout)
        layout.addWidget(analysis_frame)

    def check_winning_move(self, player):
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)             # diagonals
        ]

        for positions in win_conditions:
            values = [self.board_state[i] for i in positions]
            if values.count(player) == 2 and values.count('') == 1:
                empty_pos = positions[values.index('')]
                return True, empty_pos
        return False, -1

class AIPhilosophyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('🌌 Filosofia Erei AI - Viziunea asupra Viitorului')

        # Get screen dimensions and make dialog nearly full screen
        desktop = QDesktopWidget()
        screen = desktop.availableGeometry()
        dialog_width = min(1200, int(screen.width() * 0.9))
        dialog_height = min(900, int(screen.height() * 0.9))
        x = (screen.width() - dialog_width) // 2
        y = (screen.height() - dialog_height) // 2

        self.setGeometry(x, y, dialog_width, dialog_height)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #0f0f23, stop: 1 #1a1a2e);
                color: #eee;
            }
            QLabel {
                color: #eee;
                font-family: Arial;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QPushButton {
                background-color: #4a69bd;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3c5aa6;
            }
        """)

        layout = QVBoxLayout()

        # Title with cosmic feel
        title = QLabel('🌌 VIZIUNEA ASUPRA EREI INTELIGENȚEI ARTIFICIALE 🚀')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #00d2d3;
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                      stop: 0 #1e3c72, stop: 1 #2a5298);
            padding: 20px;
            border-radius: 12px;
            margin: 10px;
            border: 2px solid #00d2d3;
        """)
        layout.addWidget(title)

        # Create scroll area for the vision content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()

        # The philosophical journey
        self.add_programming_limits_section(scroll_layout)
        self.add_complexity_revelation_section(scroll_layout)
        self.add_neural_breakthrough_section(scroll_layout)
        self.add_ai_era_section(scroll_layout)
        self.add_applications_section(scroll_layout)
        self.add_reflection_section(scroll_layout)

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Close button with style
        close_btn = QPushButton('🧠 Înțeleg Viziunea - Înapoi la Joc')
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def add_programming_limits_section(self, layout):
        section = QFrame()
        section.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #2c3e50, stop: 1 #34495e);
                border-radius: 12px;
                padding: 15px;
                margin: 10px;
                border-left: 5px solid #e74c3c;
            }
        """)
        section_layout = QVBoxLayout()

        title = QLabel('🔧 LIMITELE PROGRAMĂRII TRADIȚIONALE')
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #e74c3c; margin-bottom: 10px;")
        section_layout.addWidget(title)

        content = QLabel("""
Pentru aceasta - rămâne strict în limitele Imaginației și Execuției programatorului să
capteze algoritmi mai profunzi.

🎯 REALITATEA PROGRAMĂRII CLASICE:
• Programatorul trebuie să anticipeze TOATE situațiile posibile
• Fiecare regulă trebuie scrisă explicit
• Complexitatea crește exponențial cu problema
• Pentru X-O: 3⁹ = 19,683 de stări posibile (încă gestionabil)
• Pentru șah: ~10⁴³ poziții posibile (imposibil de programat manual)

🧠 LIMITELE MINȚII UMANE:
Chiar și cei mai brilianti programatori sunt limitați de:
• Capacitatea de a vizualiza toate scenariile
• Timpul finit pentru a scrie toate regulile
• Imposibilitatea de a anticipa toate cazurile speciale
        """)
        content.setStyleSheet("font-size: 13px; line-height: 1.4; color: #ecf0f1;")
        content.setWordWrap(True)
        section_layout.addWidget(content)

        section.setLayout(section_layout)
        layout.addWidget(section)

    def add_complexity_revelation_section(self, layout):
        section = QFrame()
        section.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #8e44ad, stop: 1 #9b59b6);
                border-radius: 12px;
                padding: 15px;
                margin: 10px;
                border-left: 5px solid #f39c12;
            }
        """)
        section_layout = QVBoxLayout()

        title = QLabel('🌌 REVELAȚIA COMPLEXITĂȚII REALITĂȚII')
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #f39c12; margin-bottom: 10px;")
        section_layout.addWidget(title)

        content = QLabel("""
În căutarea executării sarcinilor la complexitatea nivelului uman - și confruntându-ne cu
conceptul de nerezolvat că realitatea este atât de complexă și conectată încât nu poate fi
Modelată în variabile și sisteme...

🌍 IMPOSIBILITATEA MODELĂRII COMPLETE:
• Realitatea este infinit de complexă și interconectată
• Fiecare acțiune influențează sisteme multiple
• Variabilele sunt infinite și în continuă schimbare
• Relațiile cauză-efect sunt non-liniare și chaotice

🔬 EXEMPLUL UNUI SIMPLU PIXEL:
Pentru a recunoaște un pisică într-o imagine:
• Trebuie să analizezi milioane de pixeli
• Să înțelegi formele, texturile, contextul
• Să ții cont de iluminare, unghi, ocluzie
• Să diferențiezi între pisică, câine, umbră

Cum să scrii reguli pentru TOATE aceste cazuri? IMPOSIBIL!
        """)
        content.setStyleSheet("font-size: 13px; line-height: 1.4; color: #ecf0f1;")
        content.setWordWrap(True)
        section_layout.addWidget(content)

        section.setLayout(section_layout)
        layout.addWidget(section)

    def add_neural_breakthrough_section(self, layout):
        section = QFrame()
        section.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #27ae60, stop: 1 #2ecc71);
                border-radius: 12px;
                padding: 15px;
                margin: 10px;
                border-left: 5px solid #00d2d3;
            }
        """)
        section_layout = QVBoxLayout()

        title = QLabel('🧠 DESCOPERIREA REVOLUȚIONARĂ: REȚELELE NEURALE')
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d2d3; margin-bottom: 10px;")
        section_layout.addWidget(title)

        content = QLabel("""
Am început să privim dintr-un unghi diferit - Rețelele Neurale - în loc să înțelegem
dimensiunea imposibilă a realității - să mimăm mai degrabă arhitectura creierului nostru -
și matematica profundă - permițându-i să "își urmeze" cursul (Inferență).

🔬 SCHIMBAREA DE PARADIGMĂ:
În loc să scriem reguli: "Dacă vezi formă triunghiulară + urechi ascuțite = pisică"
Creăm o rețea care învață singură: "Arată-mi 1 milion de poze cu pisici și voi învăța
să le recunosc fără să-mi spui cum"

🧮 MATEMATICA PROFUNDĂ:
• Miliarde de conexiuni neuronale artificiale
• Algoritmi de optimizare complexi (backpropagation)
• Funcții de activare non-liniare
• Gradient descent în spații multidimensionale

🌊 EMERGENȚA INTELIGENȚEI:
Prin miliarde de micro-ajustări matematice, apare ceva magic:
Comportament inteligent fără programare explicită!
        """)
        content.setStyleSheet("font-size: 13px; line-height: 1.4; color: #ecf0f1;")
        content.setWordWrap(True)
        section_layout.addWidget(content)

        section.setLayout(section_layout)
        layout.addWidget(section)

    def add_ai_era_section(self, layout):
        section = QFrame()
        section.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #e67e22, stop: 1 #f39c12);
                border-radius: 12px;
                padding: 15px;
                margin: 10px;
                border-left: 5px solid #fff;
            }
        """)
        section_layout = QVBoxLayout()

        title = QLabel('⚡ ASTFEL - MARELE CONCEPT AL PIERDERII CONTROLULUI')
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #fff; margin-bottom: 10px;")
        section_layout.addWidget(title)

        content = QLabel("""
Astfel, marele concept că nu putem fi pe deplin conștienți de ceea ce se întâmplă exact
într-o Rețea Neurală - prin urmare nu putem avea un control deplin asupra unei AI.

🌪️ PARADOXUL PUTERII AI:
• Cu cât AI devine mai puternic, cu atât devine mai misterios
• Nu știm exact DE CE ia anumite decizii
• Poate descoperi strategii pe care noi nu le înțelegem
• Poate găsi soluții la care noi nu ne-am gândit niciodată

🚀 NAȘTEREA EREI AI:
Această descoperire întâlnind progresele tehnologice actuale care ne-au oferit hardware-ul
pentru a rula aceste Modele Matematice pe el, a ridicat Era AI.

💻 REVOLUȚIA HARDWARE:
• GPU-uri puternice pentru calcule paralele
• TPU-uri specializate pentru rețele neurale
• Cloud computing la scară masivă
• Putere de calcul exponențială

🌟 REZULTATUL:
Pentru prima dată în istorie, avem atât algoritmii cât și puterea de calcul
pentru a crea inteligență artificială adevărată!
        """)
        content.setStyleSheet("font-size: 13px; line-height: 1.4; color: #ecf0f1;")
        content.setWordWrap(True)
        section_layout.addWidget(content)

        section.setLayout(section_layout)
        layout.addWidget(section)

    def add_applications_section(self, layout):
        section = QFrame()
        section.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #3498db, stop: 1 #2980b9);
                border-radius: 12px;
                padding: 15px;
                margin: 10px;
                border-left: 5px solid #ecf0f1;
            }
        """)
        section_layout = QVBoxLayout()

        title = QLabel('🚀 DINCOLO DE X-O: AUTOMATIZĂRI FĂRĂ PRECEDENT')
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #ecf0f1; margin-bottom: 10px;")
        section_layout.addWidget(title)

        content = QLabel("""
Așadar, în loc să definim logică clară și restrictivă chiar și pentru un joc X-O - am putea crea
un Model Matematic (Rețea Neurală) - și să antrenăm acest Model să învețe să joace jocul.

Desigur, aceasta merge cu mult dincolo de un Joc X-O - către Modele de Limbaj, Modele Vizuale,
și tot felul de Automatizări care nu au fost niciodată posibile înainte.

🎯 MODELE DE LIMBAJ (ChatGPT, Claude):
• Înțeleg și generează text uman
• Traduc între limbi
• Răspund la întrebări complexe
• Scriu cod, poezii, esee

👁️ MODELE VIZUALE:
• Recunosc obiecte în imagini
• Generează artă din descrieri
• Analizează imagini medicale
• Conduc mașini autonome

🎵 MODELE AUDIO:
• Recunoaștere vocală (Siri, Alexa)
• Generare de muzică
• Traducere în timp real
• Sinteză vocală naturală

🤖 AUTOMATIZĂRI IMPOSIBILE ÎNAINTE:
• Diagnostic medical automat
• Descoperire de medicamente
• Predicții meteo avansate
• Optimizare logistică globală
• Cercetare științifică automatizată
        """)
        content.setStyleSheet("font-size: 13px; line-height: 1.4; color: #ecf0f1;")
        content.setWordWrap(True)
        section_layout.addWidget(content)

        section.setLayout(section_layout)
        layout.addWidget(section)

    def add_reflection_section(self, layout):
        section = QFrame()
        section.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #1a1a2e, stop: 1 #16213e);
                border-radius: 12px;
                padding: 20px;
                margin: 10px;
                border: 3px solid #00d2d3;
            }
        """)
        section_layout = QVBoxLayout()

        title = QLabel('💭 REFLECȚIA FINALĂ: UNDE NE ÎNDREPTĂM?')
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00d2d3; margin-bottom: 15px;")
        section_layout.addWidget(title)

        content = QLabel("""
🌌 SUNTEM MARTORII UNEI REVOLUȚII:
Prin acest simplu joc X-O, am văzut diferența fundamentală între:
• Programarea tradițională (reguli explicite)
• Inteligența artificială (învățare din experiență)

🚀 VIITORUL:
• AI-ul va depăși capacitățile umane în tot mai multe domenii
• Vom colabora cu sisteme pe care nu le înțelegem complet
• Vom descoperi soluții la probleme considerate imposibile
• Vom redefini ce înseamnă să fii inteligent

🤔 ÎNTREBĂRI PROFUNDE:
• Ce se întâmplă când AI-ul devine mai inteligent decât creatorii săi?
• Cum menținem controlul asupra sistemelor pe care nu le înțelegem?
• Cum ne pregătim pentru o lume în care AI-ul poate face mai mult decât noi?

🎯 MESAJUL PENTRU GENERAȚIA URMĂTOARE:
Nu trebuie să înțelegeți fiecare calcul al unei rețele neurale.
Trebuie să înțelegeți puterea și responsabilitatea care vine cu ea.

Fiți creativi. Fiți curioși. Fiți înțelepți.
AI-ul este un instrument - folosiți-l pentru a face lumea mai bună! 🌟
        """)
        content.setStyleSheet("font-size: 14px; line-height: 1.5; color: #ecf0f1; font-weight: bold;")
        content.setWordWrap(True)
        section_layout.addWidget(content)

        section.setLayout(section_layout)
        layout.addWidget(section)

class TicTacToe(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('🎮 X și O - Învață cum "gândește" AI-ul!')
        self.setGeometry(100, 100, 700, 800)
        self.setMinimumSize(500, 600)  # Ensure minimum usable size
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #2c3e50, stop: 1 #34495e);
                color: #ecf0f1;
                font-family: Arial;
            }
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                font-size: 28px;
                font-weight: bold;
                border: 3px solid #4a627a;
                border-radius: 12px;
                min-height: 80px;
            }
            QPushButton:hover {
                background-color: #4a627a;
                border-color: #5d7aa0;
            }
            QPushButton:pressed {
                background-color: #3d566e;
            }
            #statusLabel {
                font-size: 18px;
                font-weight: bold;
                background-color: #2c3e50;
                padding: 15px;
                border-radius: 8px;
                border: 2px solid #4a627a;
            }
            #newGameButton {
                background-color: #e74c3c;
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                min-height: 40px;
            }
            #newGameButton:hover {
                background-color: #c0392b;
            }
            #explainButton {
                background-color: #9b59b6;
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                min-height: 40px;
            }
            #explainButton:hover {
                background-color: #8e44ad;
            }
            #thinkingLabel {
                font-size: 16px;
                font-weight: bold;
                color: #ffffff;
                background-color: #e67e22;
                padding: 12px;
                border-radius: 8px;
                border: 2px solid #d35400;
                margin: 5px;
            }
        """)

        self.board = [''] * 9
        self.buttons = []
        self.player_turn = True
        self.last_move_type = None
        self.last_move_pos = None

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header with philosophy button
        header_layout = QHBoxLayout()

        # Left side - welcome text
        welcome_container = QVBoxLayout()
        welcome_label = QLabel("🎯 Bun venit la X și O Educational!")
        welcome_label.setAlignment(Qt.AlignLeft)
        welcome_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #3498db; margin-bottom: 5px;")
        welcome_container.addWidget(welcome_label)

        instruction_label = QLabel("🤖 Joacă împotriva AI-ului și vezi cum 'gândește'!")
        instruction_label.setAlignment(Qt.AlignLeft)
        instruction_label.setStyleSheet("font-size: 14px; color: #bdc3c7; margin-bottom: 10px;")
        welcome_container.addWidget(instruction_label)

        header_layout.addLayout(welcome_container)
        header_layout.addStretch()  # Push button to the right

        # Right side - philosophy button
        self.philosophy_button = QPushButton('🌌 Viziunea\nErei AI')
        self.philosophy_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #5d6d7e, stop: 1 #717d8a);
                color: #ecf0f1;
                padding: 8px 12px;
                border: 1px solid #85929e;
                border-radius: 8px;
                font-size: 11px;
                font-weight: normal;
                min-width: 80px;
                max-width: 100px;
                max-height: 60px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                          stop: 0 #6c7b7f, stop: 1 #7f8c8d);
                border-color: #a6acaf;
                color: white;
            }
        """)
        self.philosophy_button.clicked.connect(self.show_philosophy)
        header_layout.addWidget(self.philosophy_button)

        main_layout.addLayout(header_layout)

        # Status label
        self.status_label = QLabel("Rândul tău (X) - Alege o poziție!", self)
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        # AI thinking indicator
        self.thinking_label = QLabel("")
        self.thinking_label.setObjectName("thinkingLabel")
        self.thinking_label.setAlignment(Qt.AlignCenter)
        self.thinking_label.setWordWrap(True)  # Allow text wrapping
        self.thinking_label.setMinimumHeight(50)  # Ensure minimum height

        # Set explicit font to ensure text renders properly
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        font.setBold(True)
        self.thinking_label.setFont(font)

        self.thinking_label.hide()
        main_layout.addWidget(self.thinking_label)

        grid_layout = QVBoxLayout()
        grid_layout.setSpacing(8)

        # Add position numbers for reference
        pos_info = QLabel("Pozițiile sunt numerotate 1-9 (stânga-dreapta, sus-jos)")
        pos_info.setAlignment(Qt.AlignCenter)
        pos_info.setStyleSheet("font-size: 12px; color: #95a5a6; margin-bottom: 5px;")
        main_layout.addWidget(pos_info)

        for i in range(3):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(8)
            for j in range(3):
                button = QPushButton('', self)
                # Remove fixed size to allow expansion
                button.setMinimumSize(80, 80)  # Minimum size to maintain usability
                button.setSizePolicy(button.sizePolicy().Expanding, button.sizePolicy().Expanding)
                position_num = i * 3 + j + 1
                button.setToolTip(f"Poziția {position_num}")
                button.clicked.connect(lambda _, b=button, index=i*3+j: self.on_button_click(b, index))
                self.buttons.append(button)
                row_layout.addWidget(button)
            grid_layout.addLayout(row_layout)

        main_layout.addLayout(grid_layout)

        # Button section
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.new_game_button = QPushButton('🔄 Joc Nou', self)
        self.new_game_button.setObjectName("newGameButton")
        self.new_game_button.clicked.connect(self.reset_game)

        self.explain_button = QPushButton('🧠 Explică ultima mutare AI', self)
        self.explain_button.setObjectName("explainButton")
        self.explain_button.clicked.connect(self.explain_last_move)
        self.explain_button.setEnabled(False)

        button_layout.addWidget(self.new_game_button)
        button_layout.addWidget(self.explain_button)
        main_layout.addLayout(button_layout)

        # Educational footer
        footer_text = "💡 AI-ul urmează 3 reguli simple: Câștigă → Blochează → Random"
        footer_label = QLabel(footer_text)
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("font-size: 12px; color: #7f8c8d; margin-top: 10px; font-style: italic;")
        main_layout.addWidget(footer_label)

        self.setLayout(main_layout)

    def on_button_click(self, button, index):
        if self.board[index] == '' and self.player_turn:
            self.board[index] = 'X'
            button.setText('X')
            button.setStyleSheet("color: #3498db; font-weight: bold;")
            self.player_turn = False
            self.status_label.setText("🤖 AI-ul se gândește...")

            if not self.check_winner():
                # Show thinking animation
                self.thinking_label.setText("🤔 Analizez tabla de joc... (3 pași)")
                self.thinking_label.show()

                # Use QTimer to simulate thinking delay
                QTimer.singleShot(1500, self.computer_move)

    def computer_move(self):
        if '' not in self.board:
            self.thinking_label.hide()
            return

        move_type = None
        chosen_move = None

        # 1. Check if computer can win
        self.thinking_label.setText("PASUL 1: Verific daca pot castiga...")
        self.thinking_label.show()
        QTimer.singleShot(1000, lambda: self.execute_step_1())

    def execute_step_1(self):
        move_type = None
        chosen_move = None

        for i in range(9):
            if self.board[i] == '':
                self.board[i] = 'O'
                if self.check_winner(silent=True) == 'O':
                    chosen_move = i
                    move_type = 'win'
                    self.board[i] = 'O'
                    self.buttons[i].setText('O')
                    self.buttons[i].setStyleSheet("color: #e74c3c; font-weight: bold;")
                    self.thinking_label.setText("PERFECT! Pot castiga la pozitia " + str(i + 1) + "!")
                    self.thinking_label.show()
                    self.finalize_move(move_type, chosen_move)
                    return
                self.board[i] = ''

        # No winning move found, check for blocking
        self.thinking_label.setText("PASUL 2: Verific daca trebuie sa blochez...")
        self.thinking_label.show()
        QTimer.singleShot(1000, lambda: self.execute_step_2())

    def execute_step_2(self):
        move_type = None
        chosen_move = None

        for i in range(9):
            if self.board[i] == '':
                self.board[i] = 'X'
                if self.check_winner(silent=True) == 'X':
                    chosen_move = i
                    move_type = 'block'
                    self.board[i] = 'O'
                    self.buttons[i].setText('O')
                    self.buttons[i].setStyleSheet("color: #e74c3c; font-weight: bold;")
                    self.thinking_label.setText("BLOCHEZ! Te opresc la pozitia " + str(i + 1) + "!")
                    self.thinking_label.show()
                    self.finalize_move(move_type, chosen_move)
                    return
                self.board[i] = ''

        # No blocking needed, make random move
        self.thinking_label.setText("PASUL 3: Aleg o pozitie la intamplare...")
        self.thinking_label.show()
        QTimer.singleShot(1000, lambda: self.execute_step_3())

    def execute_step_3(self):
        empty_cells = [i for i, val in enumerate(self.board) if val == '']
        if empty_cells:
            chosen_move = random.choice(empty_cells)
            move_type = 'random'
            self.board[chosen_move] = 'O'
            self.buttons[chosen_move].setText('O')
            self.buttons[chosen_move].setStyleSheet("color: #e74c3c; font-weight: bold;")
            self.thinking_label.setText("ALEATOR! Aleg pozitia " + str(chosen_move + 1) + " la intamplare!")
            self.thinking_label.show()
            self.finalize_move(move_type, chosen_move)

    def finalize_move(self, move_type, chosen_move):
        # Store last move info for explanation
        self.last_move_type = move_type
        self.last_move_pos = chosen_move
        self.explain_button.setEnabled(True)

        # Finish move immediately - NO TIMER HERE, user can play right away
        self.finish_computer_move()

    def finish_computer_move(self):
        # DON'T hide the thinking label - let the final message stay visible
        self.player_turn = True
        game_result = self.check_winner()
        if not game_result:
            self.status_label.setText("Rândul tău (X) - Alege o poziție!")

    def explain_last_move(self):
        if self.last_move_type and self.last_move_pos is not None:
            dialog = AIExplanationDialog(self.last_move_type, self.board, self.last_move_pos, self)
            dialog.exec_()

    def check_winner(self, silent=False):
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)             # diagonals
        ]
        for a, b, c in win_conditions:
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] != '':
                if not silent:
                    self.end_game(self.board[a])
                return self.board[a]

        if '' not in self.board:
            if not silent:
                self.end_game('Egalitate')
            return 'Egalitate'

        if not silent and self.player_turn:
            self.status_label.setText("Rândul tău (X) - Alege o poziție!")
        return None

    def end_game(self, winner):
        self.thinking_label.hide()
        if winner == 'Egalitate':
            self.status_label.setText("🤝 Egalitate! Bună încercare!")
        elif winner == 'X':
            self.status_label.setText("🎉 Felicitări! Ai câștigat!")
        else:  # winner == 'O'
            self.status_label.setText("🤖 AI-ul a câștigat! Mai încearcă!")

        for button in self.buttons:
            button.setEnabled(False)
        self.player_turn = False

        # Disable explanation button since game is over
        self.explain_button.setEnabled(False)

    def reset_game(self):
        self.board = [''] * 9
        self.player_turn = True
        self.last_move_type = None
        self.last_move_pos = None
        self.thinking_label.hide()
        self.status_label.setText("Rândul tău (X) - Alege o poziție!")
        self.explain_button.setEnabled(False)

        for button in self.buttons:
            button.setText('')
            button.setEnabled(True)
            button.setStyleSheet("")

    def show_philosophy(self):
        """
        🌌 Arată dialogul cu filosofia AI
        """
        dialog = AIPhilosophyDialog(self)
        dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = TicTacToe()
    game.show()
    sys.exit(app.exec_())