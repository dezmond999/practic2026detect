import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import random
import tempfile
import os
import deeplake
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DroneRAGSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Мультимодальная RAG-система | Детекция машин + База знаний о БПЛА")
        self.root.geometry("1400x800")
        self.root.minsize(1000, 600)
        self.root.configure(bg='#1a1a2e')
        
        self.model_yolo = None
        self.model_embedding = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        
        self.visdrone_ds = None
        self.original_image = None
        self.original_image_rgb = None
        self.detections = []
        self.hover_detection = None
        self.current_source = None
        self.tooltip_window = None
        self.show_detections_mode = True
        
        self.drone_knowledge = [
            {"id": 1, "name": "DJI Matrice 300 RTK", "description": "Профессиональный дрон для мониторинга грузовиков и автопарков. Оснащён тепловизором и ИИ-детекцией.", "specs": "Время полёта: 55 мин, Макс. скорость: 82 км/ч, Вес: 3.6 кг", "best_for": ["truck", "bus", "van"]},
            {"id": 2, "name": "Autel EVO II Pro", "description": "Отлично подходит для съёмки легковых автомобилей. Компактный складной дрон с 6K камерой.", "specs": "Время полёта: 40 мин, Макс. скорость: 72 км/ч, Вес: 1.1 кг", "best_for": ["car"]},
            {"id": 3, "name": "Parrot Anafi USA", "description": "Специализированный дрон для поиска мотоциклов и малогабаритных объектов. Компактный и складной.", "specs": "Время полёта: 32 мин, Макс. скорость: 55 км/ч, Вес: 0.5 кг", "best_for": ["motor"]},
            {"id": 4, "name": "DJI Mavic 3 Enterprise", "description": "Универсальный дрон для обнаружения любых транспортных средств. Оснащён зум-камерой и тепловизором.", "specs": "Время полёта: 45 мин, Макс. скорость: 75 км/ч, Вес: 0.9 кг", "best_for": ["car", "truck", "bus", "van", "motor"]},
            {"id": 5, "name": "DJI Mini 4 Pro", "description": "Лёгкий дрон для мониторинга легковых автомобилей и мотоциклов. Идеален для быстрых вылетов.", "specs": "Время полёта: 34 мин, Макс. скорость: 57 км/ч, Вес: 0.25 кг", "best_for": ["car", "motor"]},
            {"id": 6, "name": "Autel Dragonfish", "description": "Профессиональный дрон с вертикальным взлётом для длительного мониторинга автопарков и грузовых перевозок.", "specs": "Время полёта: 120 мин, Макс. скорость: 108 км/ч, Вес: 2.5 кг", "best_for": ["truck", "bus", "van"]}
        ]
        
        self.drone_embeddings = []
        for drone in self.drone_knowledge:
            text = f"{drone['name']} {drone['description']} {drone['specs']}"
            self.drone_embeddings.append(self.model_embedding.encode(text))
        
        self.zoom_level = 1.0
        self.zoom_min = 0.5
        self.zoom_max = 5.0
        self.pan_x, self.pan_y = 0, 0
        self.pan_start_x, self.pan_start_y = 0, 0
        self.fit_to_window = True
        
        self.status_var = tk.StringVar(value="Загрузка YOLO модели...")
        
        self.load_models()
        self.setup_ui()
        self.check_models_ready()
        self.setup_bindings()
        self.load_visdrone_dataset()
    
    def load_models(self):
        def _load():
            try:
                self.model_yolo = YOLO("best.pt")
                self.status_var.set("Модель готова")
            except Exception as e:
                self.status_var.set(f"Ошибка YOLO: {str(e)[:50]}...")
        threading.Thread(target=_load, daemon=True).start()
    
    def load_visdrone_dataset(self):
        def _load():
            try:
                self.visdrone_ds = deeplake.load('hub://activeloop/visdrone-det-train')
                print("VisDrone датасет готов")
            except Exception as e:
                self.visdrone_ds = None
        threading.Thread(target=_load, daemon=True).start()
    
    def check_models_ready(self):
        if self.model_yolo is not None:
            return
        self.root.after(500, self.check_models_ready)
    
    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_panel = tk.Frame(main_frame, bg='#16213e', width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        right_panel = tk.Frame(main_frame, bg='#0f3460')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(left_panel, text="MULTIMODAL RAG SYSTEM", 
                font=('Arial', 14, 'bold'), bg='#16213e', fg='#e94560').pack(pady=15)
        tk.Label(left_panel, text="Детекция машин + База знаний о БПЛА", 
                font=('Arial', 9), bg='#16213e', fg='#aaa').pack(pady=(0, 15))
        
        tk.Label(left_panel, text="1. ЗАГРУЗИТЕ ИЗОБРАЖЕНИЕ", 
                font=('Arial', 10, 'bold'), bg='#16213e', fg='#e94560').pack(anchor='w', padx=20, pady=(10,5))
        
        btn_frame = tk.Frame(left_panel, bg='#16213e')
        btn_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Button(btn_frame, text="ЗАГРУЗИТЬ ФОТО", 
                 font=('Arial', 10, 'bold'), bg='#e94560', fg='white',
                 command=self.load_image, height=1).pack(fill=tk.X, pady=2)
        
        tk.Button(btn_frame, text="СЛУЧАЙНОЕ ФОТО ИЗ VISDRONE", 
                 font=('Arial', 10, 'bold'), bg='#2c3e66', fg='white',
                 command=self.load_random_visdrone, height=1).pack(fill=tk.X, pady=2)
        
        tk.Label(left_panel, text="2. ДЕТЕКЦИЯ МАШИН", 
                font=('Arial', 10, 'bold'), bg='#16213e', fg='#e94560').pack(anchor='w', padx=20, pady=(15,5))
        
        params_frame = tk.Frame(left_panel, bg='#16213e')
        params_frame.pack(fill=tk.X, padx=20, pady=5)
        
        conf_frame = tk.Frame(params_frame, bg='#16213e')
        conf_frame.pack(fill=tk.X, pady=5)
        tk.Label(conf_frame, text="Порог уверенности:", bg='#16213e', fg='white').pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=0.25)
        tk.Scale(conf_frame, from_=0.1, to=0.5, resolution=0.05,
                orient=tk.HORIZONTAL, variable=self.conf_var,
                bg='#16213e', fg='white', highlightthickness=0).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        self.tiling_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(params_frame, text="Использовать тайлинг (для мелких машин)",
                      variable=self.tiling_enabled, bg='#16213e', fg='white',
                      selectcolor='#16213e').pack(anchor='w', pady=5)
        
        tile_frame = tk.Frame(params_frame, bg='#16213e')
        tile_frame.pack(fill=tk.X, pady=5)
        tk.Label(tile_frame, text="Размер тайла:", bg='#16213e', fg='white').pack(side=tk.LEFT)
        self.tile_var = tk.IntVar(value=640)
        self.tile_menu = ttk.Combobox(tile_frame, textvariable=self.tile_var, 
                                      values=[640, 800, 1024, 1280], width=8, state='readonly')
        self.tile_menu.pack(side=tk.RIGHT, padx=5)
        
        overlap_frame = tk.Frame(params_frame, bg='#16213e')
        overlap_frame.pack(fill=tk.X, pady=5)
        tk.Label(overlap_frame, text="Перекрытие тайлов:", bg='#16213e', fg='white').pack(side=tk.LEFT)
        self.overlap_var = tk.DoubleVar(value=0.2)
        tk.Scale(overlap_frame, from_=0.1, to=0.4, resolution=0.05,
                orient=tk.HORIZONTAL, variable=self.overlap_var,
                bg='#16213e', fg='white', highlightthickness=0).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        detect_frame = tk.Frame(left_panel, bg='#16213e')
        detect_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.detect_btn = tk.Button(detect_frame, text="НАЙТИ МАШИНЫ", 
                                   font=('Arial', 11, 'bold'), bg='#e94560', fg='white',
                                   command=self.detect_vehicles, height=1, state='disabled')
        self.detect_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.original_btn = tk.Button(detect_frame, text="ПОКАЗАТЬ ОРИГИНАЛ", 
                                     font=('Arial', 11, 'bold'), bg='#1a5c5e', fg='white',
                                     command=self.show_original, height=1, state='disabled')
        self.original_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        tk.Label(left_panel, text="3. ЗАПРОС К БАЗЕ ЗНАНИЙ О БПЛА", 
                font=('Arial', 10, 'bold'), bg='#16213e', fg='#e94560').pack(anchor='w', padx=20, pady=(15,5))
        
        self.query_text = tk.Text(left_panel, height=3, width=35, bg='#0f3460', fg='white',
                                  font=('Arial', 9), wrap=tk.WORD)
        self.query_text.pack(fill=tk.X, padx=20, pady=5)
        self.query_text.insert("1.0", "Например: какой дрон лучше для мониторинга грузовиков?")
        
        query_frame = tk.Frame(left_panel, bg='#16213e')
        query_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Button(query_frame, text="ПОИСК В БАЗЕ БПЛА", 
                 font=('Arial', 10, 'bold'), bg='#2c3e66', fg='white',
                 command=self.rag_search).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        tk.Button(query_frame, text="ПО РЕЗУЛЬТАТАМ ДЕТЕКЦИИ", 
                 font=('Arial', 10, 'bold'), bg='#1a5c5e', fg='white',
                 command=self.rag_search_by_detection).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        tk.Label(left_panel, text="РЕЗУЛЬТАТЫ RAG", 
                font=('Arial', 10, 'bold'), bg='#16213e', fg='#e94560').pack(anchor='w', padx=20, pady=(15,5))
        
        self.rag_result = tk.Text(left_panel, height=8, bg='#0f3460', fg='#ffd700',
                                  font=('Arial', 9), relief=tk.FLAT, wrap=tk.WORD, state=tk.DISABLED)
        self.rag_result.pack(fill=tk.X, padx=20, pady=5)
        self.rag_result.insert("1.0", "Результаты поиска появятся здесь...")
        
        tk.Label(left_panel, textvariable=self.status_var, bg='#16213e', 
                fg='#aaa', font=('Arial', 8)).pack(side=tk.BOTTOM, pady=10)
        
        canvas_container = tk.Frame(right_panel, bg='#0f3460')
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        h_scroll = tk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        v_scroll = tk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        
        self.canvas = tk.Canvas(canvas_container, bg='#0f3460', highlightthickness=0,
                                xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        stats_frame = tk.Frame(right_panel, bg='#1a1a2e', bd=2, relief=tk.RIDGE)
        stats_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))
        
        tk.Label(stats_frame, text="СТАТИСТИКА ДЕТЕКЦИИ", 
                font=('Arial', 9, 'bold'), bg='#1a1a2e', fg='#e94560').pack(anchor='w', padx=5, pady=2)
        
        self.stats_var = tk.StringVar(value="Ожидание загрузки...")
        tk.Label(stats_frame, textvariable=self.stats_var, bg='#1a1a2e', fg='#00ff00',
                font=('Consolas', 10, 'bold'), anchor='w', justify=tk.LEFT).pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(right_panel, text="Управление: Ctrl+колесо - зум | Правая кнопка - панорамирование | Наведите курсор на машину — появится подсказка",
                bg='#0f3460', fg='#aaa', font=('Arial', 9)).pack(side=tk.BOTTOM, pady=5)
    
    def setup_bindings(self):
        self.root.bind("<Control-plus>", lambda e: self.zoom(1.25))
        self.root.bind("<Control-minus>", lambda e: self.zoom(0.8))
        self.root.bind("<Control-0>", lambda e: self.reset_view())
        
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.hide_tooltip)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.do_pan)
    
    def show_original(self):
        if self.original_image_rgb is None:
            return
        self.show_detections_mode = not self.show_detections_mode
        self.update_display()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.current_source = f"Пользовательское фото: {os.path.basename(file_path)}"
            self.load_and_process(file_path)
    
    def load_random_visdrone(self):
        if self.visdrone_ds is None:
            messagebox.showwarning("Ошибка", "VisDrone датасет ещё не загружен.")
            return
        
        self.status_var.set("Загрузка случайного фото из VisDrone...")
        
        def _load():
            try:
                idx = random.randint(0, len(self.visdrone_ds) - 1)
                sample = self.visdrone_ds[idx]
                img_array = sample['images'].numpy()
                
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, f"visdrone_{idx}.jpg")
                cv2.imwrite(temp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                
                self.current_source = f"VisDrone датасет (изображение #{idx})"
                self.load_and_process(temp_path)
            except Exception as e:
                self.status_var.set(f"Ошибка VisDrone: {str(e)[:50]}")
        
        threading.Thread(target=_load, daemon=True).start()
    
    def load_and_process(self, path):
        self.original_image = cv2.imread(path)
        if self.original_image is None:
            self.status_var.set("Ошибка загрузки")
            return
        
        self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.detections = []
        self.hover_detection = None
        self.show_detections_mode = True
        self.fit_to_window = True
        self.reset_view()
        
        self.detect_btn.config(state='normal')
        self.original_btn.config(state='normal')
        self.status_var.set(f"Загружено: {self.current_source}")
        
        self.update_stats()
        self.update_display()
    
    def detect_vehicles(self):
        if self.original_image is None:
            messagebox.showwarning("Ошибка", "Сначала загрузите фото!")
            return
        
        self.detect_btn.config(state='disabled', text='ПОИСК...')
        self.status_var.set("Поиск машин...")
        
        def _detect():
            try:
                if self.tiling_enabled.get():
                    detections = self.run_detection_with_tiling(
                        self.original_image,
                        tile_size=self.tile_var.get(),
                        overlap=self.overlap_var.get(),
                        conf_threshold=self.conf_var.get()
                    )
                else:
                    detections = self.run_detection_simple(self.original_image, conf_threshold=self.conf_var.get())
                
                self.detections = detections
                self.show_detections_mode = True
                self.update_stats()
                self.update_display()
                
                if detections:
                    self.status_var.set(f"Найдено машин: {len(detections)}")
                else:
                    self.status_var.set("Машин не найдено. Попробуйте понизить порог")
            except Exception as e:
                self.status_var.set(f"Ошибка: {str(e)[:80]}")
            self.detect_btn.config(state='normal', text='НАЙТИ МАШИНЫ')
        
        threading.Thread(target=_detect, daemon=True).start()
    
    def run_detection_simple(self, img, conf_threshold=0.25):
        results = self.model_yolo(img, conf=conf_threshold, iou=0.45, verbose=False)
        
        vehicle_classes = {3: "car", 4: "van", 5: "truck", 8: "bus", 9: "motor"}
        colors = {"car": (0,255,0), "van": (100,255,100), "truck": (0,165,255), "bus": (0,0,255), "motor": (255,0,255)}
        
        detections = []
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in vehicle_classes:
                        continue
                    
                    label = vehicle_classes[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'label': label,
                        'confidence': conf,
                        'color': colors[label]
                    })
        return detections
    
    def run_detection_with_tiling(self, img, tile_size=640, overlap=0.2, conf_threshold=0.25):
        h, w = img.shape[:2]
        stride = int(tile_size * (1 - overlap))
        
        vehicle_classes = {3: "car", 4: "van", 5: "truck", 8: "bus", 9: "motor"}
        colors = {"car": (0,255,0), "van": (100,255,100), "truck": (0,165,255), "bus": (0,0,255), "motor": (255,0,255)}
        
        if h <= tile_size and w <= tile_size:
            tiles = [(0, 0, w, h)]
        else:
            tiles = []
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
                    if (x2 - x) >= tile_size//2 and (y2 - y) >= tile_size//2:
                        tiles.append((x, y, x2, y2))
        
        all_detections = []
        for x1, y1, x2, y2 in tiles:
            tile = img[y1:y2, x1:x2]
            results = self.model_yolo(tile, conf=conf_threshold, iou=0.45, verbose=False)
            
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        if cls_id not in vehicle_classes:
                            continue
                        
                        label = vehicle_classes[cls_id]
                        x1_box, y1_box, x2_box, y2_box = map(float, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        
                        gx1, gy1 = int(x1 + x1_box), int(y1 + y1_box)
                        gx2, gy2 = int(x1 + x2_box), int(y1 + y2_box)
                        
                        all_detections.append({
                            'bbox': (gx1, gy1, gx2, gy2),
                            'label': label,
                            'confidence': conf,
                            'color': colors[label]
                        })
        
        return self.nms(all_detections)
    
    def nms(self, detections, iou_threshold=0.5):
        if not detections:
            return []
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        keep = []
        
        for det in detections:
            keep_flag = True
            x1, y1, x2, y2 = det['bbox']
            for kept in keep:
                kx1, ky1, kx2, ky2 = kept['bbox']
                ix1 = max(x1, kx1)
                iy1 = max(y1, ky1)
                ix2 = min(x2, kx2)
                iy2 = min(y2, ky2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (kx2 - kx1) * (ky2 - ky1)
                    iou = inter / (area1 + area2 - inter)
                    if iou > iou_threshold:
                        keep_flag = False
                        break
            if keep_flag:
                keep.append(det)
        return keep
    
    def rag_search(self):
        query = self.query_text.get("1.0", tk.END).strip()
        if not query:
            self.rag_result.config(state=tk.NORMAL)
            self.rag_result.delete("1.0", tk.END)
            self.rag_result.insert("1.0", "Введите запрос!")
            self.rag_result.config(state=tk.DISABLED)
            return
        
        self.status_var.set("Поиск в базе знаний о БПЛА...")
        self.rag_result.config(state=tk.NORMAL)
        self.rag_result.delete("1.0", tk.END)
        
        query_embedding = self.model_embedding.encode(query)
        similarities = cosine_similarity([query_embedding], self.drone_embeddings)[0]
        best_indices = np.argsort(similarities)[::-1][:3]
        
        result = f"По запросу: '{query}'\n\n"
        result += "=" * 50 + "\n\n"
        
        for i, idx in enumerate(best_indices):
            drone = self.drone_knowledge[idx]
            score = similarities[idx]
            result += f"{i+1}. {drone['name']} (релевантность: {score:.2%})\n"
            result += f"   {drone['description']}\n"
            result += f"   {drone['specs']}\n"
            result += f"   Лучше всего для: {', '.join(drone['best_for'])}\n\n"
        
        self.rag_result.insert("1.0", result)
        self.rag_result.config(state=tk.DISABLED)
        self.status_var.set("Поиск завершён")
    
    def rag_search_by_detection(self):
        if not self.detections:
            self.rag_result.config(state=tk.NORMAL)
            self.rag_result.delete("1.0", tk.END)
            self.rag_result.insert("1.0", "Сначала выполните детекцию машин!")
            self.rag_result.config(state=tk.DISABLED)
            return
        
        vehicle_types = list(set([d['label'] for d in self.detections]))
        
        if not vehicle_types:
            self.rag_result.config(state=tk.NORMAL)
            self.rag_result.delete("1.0", tk.END)
            self.rag_result.insert("1.0", "Машины не найдены. Попробуйте понизить порог уверенности.")
            self.rag_result.config(state=tk.DISABLED)
            return
        
        query = f"какой дрон лучше для мониторинга {', '.join(vehicle_types)}"
        
        self.status_var.set("Поиск дронов по результатам детекции...")
        self.query_text.delete("1.0", tk.END)
        self.query_text.insert("1.0", query)
        
        query_embedding = self.model_embedding.encode(query)
        similarities = cosine_similarity([query_embedding], self.drone_embeddings)[0]
        best_indices = np.argsort(similarities)[::-1][:3]
        
        result = f"ПО РЕЗУЛЬТАТАМ ДЕТЕКЦИИ\n\n"
        result += f"Обнаружены типы машин: {', '.join(vehicle_types)}\n"
        result += "=" * 50 + "\n\n"
        result += "Рекомендуемые БПЛА:\n\n"
        
        for i, idx in enumerate(best_indices):
            drone = self.drone_knowledge[idx]
            score = similarities[idx]
            result += f"{i+1}. {drone['name']} (совместимость: {score:.2%})\n"
            result += f"   {drone['description']}\n"
            result += f"   {drone['specs']}\n\n"
        
        self.rag_result.config(state=tk.NORMAL)
        self.rag_result.delete("1.0", tk.END)
        self.rag_result.insert("1.0", result)
        self.rag_result.config(state=tk.DISABLED)
        self.status_var.set("Рекомендации по БПЛА готовы")
    
    def update_stats(self):
        if not self.detections:
            self.stats_var.set("Машины не найдены")
            return
        
        cars = sum(1 for d in self.detections if d['label'] == 'car')
        vans = sum(1 for d in self.detections if d['label'] == 'van')
        trucks = sum(1 for d in self.detections if d['label'] == 'truck')
        buses = sum(1 for d in self.detections if d['label'] == 'bus')
        motors = sum(1 for d in self.detections if d['label'] == 'motor')
        total = len(self.detections)
        
        stats = f"Легковых:{cars}  Фургонов:{vans}  Грузовиков:{trucks}  Автобусов:{buses}  Мотоциклов:{motors}  Всего:{total}"
        self.stats_var.set(stats)
    
    def update_display(self):
        if self.original_image_rgb is None:
            return
        
        h, w = self.original_image_rgb.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if self.fit_to_window and canvas_w > 1 and canvas_h > 1:
            scale_w = canvas_w / w
            scale_h = canvas_h / h
            self.zoom_level = min(scale_w, scale_h) * 0.95
            self.pan_x = 0
            self.pan_y = 0
            self.fit_to_window = False
        
        new_w, new_h = int(w * self.zoom_level), int(h * self.zoom_level)
        
        if self.show_detections_mode and self.detections:
            display = cv2.resize(self.original_image_rgb, (new_w, new_h))
            for det in self.detections:
                x1, y1, x2, y2 = det['bbox']
                x1, y1 = int(x1 * self.zoom_level), int(y1 * self.zoom_level)
                x2, y2 = int(x2 * self.zoom_level), int(y2 * self.zoom_level)
                
                color = det['color']
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                text = f"{det['label']} {det['confidence']:.2f}"
                cv2.putText(display, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if self.hover_detection == det:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 3)
        else:
            display = cv2.resize(self.original_image_rgb, (new_w, new_h))
        
        self.current_photo = ImageTk.PhotoImage(Image.fromarray(display))
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.canvas.create_image(self.pan_x, self.pan_y, anchor='nw', image=self.current_photo)
    
    def show_tooltip(self, x, y, text):
        self.hide_tooltip()
        self.tooltip_window = tk.Toplevel(self.root)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x+15}+{y+10}")
        self.tooltip_window.configure(bg='#1a1a2e')
        
        frame = tk.Frame(self.tooltip_window, bg='#e94560', padx=1, pady=1)
        frame.pack()
        inner = tk.Frame(frame, bg='#16213e', padx=10, pady=5)
        inner.pack()
        
        tk.Label(inner, text="Авто", font=('Arial', 14), bg='#16213e', fg='#e94560').pack(side=tk.LEFT, padx=(0,5))
        tk.Label(inner, text=text, font=('Arial', 9, 'bold'), bg='#16213e', fg='#ffd700', justify=tk.LEFT).pack(side=tk.LEFT)
        
        self.tooltip_id = self.root.after(2000, self.hide_tooltip)
    
    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
        if hasattr(self, 'tooltip_id'):
            self.root.after_cancel(self.tooltip_id)
    
    def on_mouse_move(self, event):
        if not self.detections or not self.show_detections_mode:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        img_x = int((canvas_x - self.pan_x) / self.zoom_level)
        img_y = int((canvas_y - self.pan_y) / self.zoom_level)
        
        found = None
        for det in self.detections:
            x1, y1, x2, y2 = det['bbox']
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                found = det
                break
        
        if found != self.hover_detection:
            self.hover_detection = found
            self.update_display()
            if found:
                ru_names = {"car": "легковой автомобиль", "van": "фургон", "truck": "грузовик", "bus": "автобус", "motor": "мотоцикл"}
                ru_name = ru_names.get(found['label'], found['label'])
                tooltip_text = f"{ru_name.upper()}\nТип: {found['label']}\nТочность: {found['confidence']:.1%}"
                self.show_tooltip(event.x_root, event.y_root, tooltip_text)
            else:
                self.hide_tooltip()
    
    def on_mousewheel(self, event):
        if event.state & 0x0004:
            if event.delta > 0 or event.num == 4:
                self.zoom(1.25)
            else:
                self.zoom(0.8)
        else:
            self.canvas.yview_scroll(int(-event.delta / 120), "units")
    
    def zoom(self, factor):
        self.fit_to_window = False
        new_zoom = self.zoom_level * factor
        if self.zoom_min <= new_zoom <= self.zoom_max:
            self.zoom_level = new_zoom
            self.update_display()
    
    def reset_view(self):
        self.fit_to_window = True
        self.zoom_level = 1.0
        self.pan_x, self.pan_y = 0, 0
        self.update_display()
    
    def start_pan(self, event):
        self.pan_start_x, self.pan_start_y = event.x, event.y
    
    def do_pan(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.pan_x += dx
        self.pan_y += dy
        self.pan_start_x, self.pan_start_y = event.x, event.y
        self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = DroneRAGSystem(root)
    root.mainloop()
