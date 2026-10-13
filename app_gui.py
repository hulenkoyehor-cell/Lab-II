import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np

from perceptron import RosenblattPerceptron
from hopfield import HopfieldNetwork
from hamming import HammingNetwork

class SimpleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Лабораторна №2 — Нейронні мережі")
        self.root.geometry("950x650")

        self.SIZE = 64  
        self.perceptron = RosenblattPerceptron(self.SIZE)
        self.hopfield = HopfieldNetwork(self.SIZE)
        self.hamming = None

        self.patterns_t2 = [None, None, None]
        self.current_t1 = None
        self.current_t2 = None
        self.img_cache = {}  

        # Вкладки
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill=tk.BOTH, expand=True)

        self.t1 = ttk.Frame(tabs)
        self.t2 = ttk.Frame(tabs)

        tabs.add(self.t1, text="1. Перцептрон (2 класи)")
        tabs.add(self.t2, text="2. Хопфілд та Хеммінг (3 класи)")

        self._init_tab1()
        self._init_tab2()

    def _read_image(self, path):
        # картинка стискаєм до 8х8
        img = Image.open(path).convert('L')
        img_resized = img.resize((8, 8), Image.Resampling.NEAREST)
        abs_vec = np.array(img_resized).flatten()
        norm_vec = np.where(abs_vec > 128, 1, -1)
        
        
        preview = ImageTk.PhotoImage(img.resize((70, 70)))
        return norm_vec, abs_vec, preview

    # 1 вкладка
    def _init_tab1(self):
        f = ttk.Frame(self.t1, padding=10)
        f.pack(fill=tk.BOTH, expand=True)

        ctrl = ttk.Frame(f)
        ctrl.pack(fill=tk.X, pady=5)

        ttk.Button(ctrl, text="Завантажити образ", command=self._load_t1).pack(side=tk.LEFT, padx=5)
        ttk.Label(ctrl, text="Клас (d):").pack(side=tk.LEFT, padx=2)
        
        self.d_val = ttk.Combobox(ctrl, values=["1", "-1"], width=3, state="readonly")
        self.d_val.set("1")
        self.d_val.pack(side=tk.LEFT, padx=5)

        ttk.Button(ctrl, text="Перевірити", command=self._pred_t1).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="Навчити (корекція)", command=self._train_t1).pack(side=tk.LEFT, padx=5)

        # показує картинку
        self.lbl_img1 = ttk.Label(f, text="[Немає фото]", relief=tk.SOLID)
        self.lbl_img1.pack(pady=5)

        self.txt1 = tk.Text(f, height=18, width=90)
        self.txt1.pack(fill=tk.BOTH, expand=True)

    def _load_t1(self):
        p = filedialog.askopenfilename()
        if not p: return
        self.current_t1, abs_v, preview = self._read_image(p)
        self.img_cache['t1'] = preview
        self.lbl_img1.config(image=preview, text="")
        
        self.txt1.insert(tk.END, f"\nЗавантажено {p.split('/')[-1]}:\n")
        self.txt1.insert(tk.END, f"Абсолютний вектор (перші 8): {abs_v[:8]}\n")
        self.txt1.insert(tk.END, f"Нормований вектор (1/-1): {self.current_t1[:10]}...\n")

    def _pred_t1(self):
        if self.current_t1 is None: return
        res = self.perceptron.predict(self.current_t1)
        self.txt1.insert(tk.END, f"Перцептрон каже: Клас {res}\n")

    def _train_t1(self):
        if self.current_t1 is None: return
        d = int(self.d_val.get())
        _, msg = self.perceptron.train_step(self.current_t1, d)
        self.txt1.insert(tk.END, f"[Результат навчання]: {msg}\n")

    # вкладка 2
    def _init_tab2(self):
        f = ttk.Frame(self.t2, padding=10)
        f.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(f)
        top_frame.pack(fill=tk.X, pady=5)

        self.pat_lbls = []
        for i in range(3):
            col_frame = ttk.Frame(top_frame)
            col_frame.pack(side=tk.LEFT, padx=15)
            
            ttk.Button(col_frame, text=f"Еталон {i+1}", command=lambda idx=i: self._load_pat(idx)).pack()
            lbl = ttk.Label(col_frame, text="[Порожньо]", relief=tk.SOLID)
            lbl.pack(pady=2)
            self.pat_lbls.append(lbl)

        ttk.Button(top_frame, text="Навчити мережі", command=self._train_t2).pack(side=tk.LEFT, padx=10)

        mid_frame = ttk.Frame(f)
        mid_frame.pack(fill=tk.X, pady=10)

        ttk.Button(mid_frame, text="Тестова картинка", command=self._load_t2).pack(side=tk.LEFT, padx=5)
        self.lbl_img2 = ttk.Label(mid_frame, text="[Немає фото]", relief=tk.SOLID)
        self.lbl_img2.pack(side=tk.LEFT, padx=5)

        ttk.Button(mid_frame, text="Тест Хопфілдом", command=self._run_hopfield).pack(side=tk.LEFT, padx=5)
        ttk.Button(mid_frame, text="Тест Хеммінгом", command=self._run_hamming).pack(side=tk.LEFT, padx=5)

        self.txt2 = tk.Text(f, height=18, width=90)
        self.txt2.pack(fill=tk.BOTH, expand=True)

    def _load_pat(self, idx):
        p = filedialog.askopenfilename()
        if not p: return
        self.patterns_t2[idx], _, preview = self._read_image(p)
        self.img_cache[f'pat_{idx}'] = preview
        self.pat_lbls[idx].config(image=preview, text="")
        self.txt2.insert(tk.END, f"Завантажено Еталон {idx+1}\n")

    def _train_t2(self):
        if any(p is None for p in self.patterns_t2):
            messagebox.showwarning("Увага", "Завантажте всі 3 еталони!")
            return
        pats = np.array(self.patterns_t2)
        self.hopfield.train(pats)
        self.hamming = HammingNetwork(pats)
        
        self.txt2.insert(tk.END, "\nМережі Хопфілда та Хеммінга успішно навчено!\n")
        self.txt2.insert(tk.END, f"Фрагмент матриці ваг Хопфілда W (3x3):\n{np.round(self.hopfield.W[:3,:3], 2)}\n\n")

    def _load_t2(self):
        p = filedialog.askopenfilename()
        if not p: return
        self.current_t2, _, preview = self._read_image(p)
        self.img_cache['t2'] = preview
        self.lbl_img2.config(image=preview, text="")
        self.txt2.insert(tk.END, "Завантажено тестову картинку.\n")

    def _run_hopfield(self):
        if self.current_t2 is None or self.hopfield is None: return
        res, iters = self.hopfield.predict(self.current_t2)
        
        match = "Невідомий образ"
        for i, p in enumerate(self.patterns_t2):
            if np.array_equal(res, p): match = f"Еталон №{i+1}"; break
            
        self.txt2.insert(tk.END, f"[Хопфілд]: Згадав за {iters} ітерацій. Це -> {match}\n")

    def _run_hamming(self):
        if self.current_t2 is None or self.hamming is None: return
        winner, y1 = self.hamming.predict(self.current_t2)
        self.txt2.insert(tk.END, f"[Хеммінг]: Розпізнано Еталон №{winner+1} (Активність шару: {np.round(y1, 2)})\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleApp(root)
    root.mainloop()