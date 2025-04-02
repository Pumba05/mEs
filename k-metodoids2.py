import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from math import sqrt
import threading # Adăugat pentru a nu bloca GUI-ul
import queue     # Adăugat pentru comunicare între thread-uri

# --- Funcții Utilitare (majoritatea nemodificate) ---

def load_points(filename):
    points = []
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                processedline = line.replace(',','')
                parts = processedline.strip().split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        points.append(np.array([x, y]))
                    except ValueError:
                        print(f"Avertisment: Linie invalidă ({line_num}) în fișier, sărită: {line.strip()}")
                elif line.strip(): # Ignoră liniile complet goale, dar avertizează pentru altele
                     print(f"Avertisment: Linie ({line_num}) cu format neașteptat, sărită: {line.strip()}")

    except FileNotFoundError:
        messagebox.showerror("Eroare Fișier", f"Fișierul '{filename}' nu a fost găsit.")
        return None
    except Exception as e:
        messagebox.showerror("Eroare Citire", f"Eroare la citirea fișierului: {e}")
        return None
        
    if not points:
        messagebox.showwarning("Fișier Gol", "Nu s-au putut încărca puncte valide din fișier.")
        return None
        
    print(f"Loaded {len(points)} points from {filename}.")
    return np.array(points)

def calculate_distance(point1, point2, method='euclidean'):
    if method == 'euclidean':
        return np.linalg.norm(point1 - point2) 
    elif method == 'manhattan':
        return np.sum(np.abs(point1 - point2))
    else:
        # Acest caz nu ar trebui să apară cu Combobox, dar e bine să fie aici
        raise ValueError("Metodă de distanță necunoscută.")

def calculate_convergence_error(points, medoids, assignments, distance_method):
    total_error = 0
    k = len(medoids)
    if k == 0 or assignments is None: # Verificare suplimentară
        return 0
        
    for i in range(k):
        # Asigură-te că assignments are lungimea corectă și conține indecși valizi
        if points is None or assignments is None or len(assignments) != len(points):
             print("Eroare: Datele punctelor sau asignările sunt invalide în calculul erorii.")
             continue # Sau tratează eroarea altfel

        try:
            cluster_points_indices = np.where(assignments == i)[0]
            if len(cluster_points_indices) > 0:
                 cluster_points = points[cluster_points_indices]
                 medoid = medoids[i]
                 distances = [calculate_distance(p, medoid, distance_method) for p in cluster_points]
                 total_error += np.sum(distances)
        except IndexError as e:
             print(f"Eroare de indexare în calculul erorii pentru clusterul {i}: {e}")
             print(f"  Lungime assignments: {len(assignments)}, index încercat: {i}")
             print(f"  Lungime medoids: {len(medoids)}")
             # Poate ar trebui să oprești sau să continui cu precauție
             continue
             
    return total_error

# --- Funcția K-Medoids (modificată pentru GUI și raportare progres) ---

def k_medoids_worker(points, k, distance_method, max_epochs, tolerance, progress_queue):
    """Rulează algoritmul K-Medoids și trimite actualizări prin coadă."""
    try:
        n_points = len(points)
        if k > n_points:
            progress_queue.put({"status": "error", "message": "k nu poate fi mai mare decât numărul de puncte"})
            return

        # 1. Inițializare Medoizi
        initial_indices = random.sample(range(n_points), k)
        medoids = points[initial_indices]
        progress_queue.put({"status": "log", "message": f"Medoizi initiali (indecși): {initial_indices}"})
        
        assignments = np.zeros(n_points, dtype=int)
        
        # Trimite starea inițială pentru plotare
        for i in range(n_points):
            distances = [calculate_distance(points[i], m, distance_method) for m in medoids]
            assignments[i] = np.argmin(distances)
        progress_queue.put({
            "status": "plot_update", 
            "epoch": 0, 
            "medoids": np.copy(medoids), 
            "assignments": np.copy(assignments),
            "title": "Stare Initiala"
        })

        last_error = float('inf')
        
        for epoch in range(1, max_epochs + 1):
            progress_queue.put({"status": "log", "message": f"\n--- Epoca {epoch} ---"})
            
            # 2. Asignare puncte
            changed_assignments = False
            # Optimizare: Calculează toate distanțele o dată dacă e posibil
            # distances_to_medoids = np.array([[calculate_distance(p, m, distance_method) for m in medoids] for p in points])
            # new_assignments = np.argmin(distances_to_medoids, axis=1)
            new_assignments = np.copy(assignments) # Pornește de la asignările vechi
            for i in range(n_points):
                 distances = [calculate_distance(points[i], m, distance_method) for m in medoids]
                 closest_medoid_index = np.argmin(distances)
                 if assignments[i] != closest_medoid_index:
                    new_assignments[i] = closest_medoid_index
                    changed_assignments = True

            assignments = new_assignments # Actualizează toate asignările după ce ai iterat prin toate punctele

            # 4. Actualizare Medoizi
            new_medoids = np.copy(medoids)
            medoid_changed = False
            for i in range(k):
                cluster_points_indices = np.where(assignments == i)[0]
                
                if len(cluster_points_indices) == 0:
                    progress_queue.put({"status": "log", "message": f"Atenție: Medoidul {i} nu are puncte asignate."})
                    continue 

                cluster_points = points[cluster_points_indices]
                
                # Găsește punctul din cluster cel mai apropiat de centrul de greutate
                gravity_center = np.mean(cluster_points, axis=0)
                min_dist_to_gravity = float('inf')
                new_medoid_index_in_cluster = -1

                for idx, point_in_cluster in enumerate(cluster_points):
                    dist = calculate_distance(point_in_cluster, gravity_center, distance_method)
                    if dist < min_dist_to_gravity:
                        min_dist_to_gravity = dist
                        new_medoid_index_in_cluster = idx 

                if new_medoid_index_in_cluster != -1:
                    potential_new_medoid = cluster_points[new_medoid_index_in_cluster]
                    if not np.array_equal(new_medoids[i], potential_new_medoid):
                        new_medoids[i] = potential_new_medoid
                        medoid_changed = True
                        # progress_queue.put({"status": "log", "message": f"Medoidul {i} mutat."}) # Poate fi prea mult log

            medoids = new_medoids

            # 5. Calculează eroarea
            current_error = calculate_convergence_error(points, medoids, assignments, distance_method)
            progress_queue.put({"status": "log", "message": f"Eroare totală: {current_error:.4f}"})

            # Trimite actualizare pentru plotare (poate opțional, la fiecare N epoci)
            if epoch % 5 == 0 or medoid_changed or epoch == 1: # Ex: plot la fiecare 5 epoci sau dacă s-a schimbat ceva
                 progress_queue.put({
                    "status": "plot_update", 
                    "epoch": epoch, 
                    "medoids": np.copy(medoids), 
                    "assignments": np.copy(assignments),
                    "title": f"Actualizare Epoca {epoch}"
                 })

            # 6. Verifică convergența
            error_difference = abs(last_error - current_error)
            if not medoid_changed or error_difference < tolerance:
                progress_queue.put({"status": "log", "message": f"\nAlgoritmul a convergat după {epoch} epoci."})
                progress_queue.put({"status": "finished", "final_medoids": medoids, "final_assignments": assignments, "epoch": epoch})
                return
            
            last_error = current_error

        # Dacă a ieșit din for din cauza max_epochs
        progress_queue.put({"status": "log", "message": f"\nAlgoritmul s-a oprit după numărul maxim de epoci ({max_epochs})."})
        progress_queue.put({"status": "finished", "final_medoids": medoids, "final_assignments": assignments, "epoch": max_epochs})

    except Exception as e:
        progress_queue.put({"status": "error", "message": f"Eroare în timpul execuției K-Medoids: {e}"})


# --- Clasa principală a aplicației GUI ---

class KMedoidsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("K-Medoids GUI")
        self.geometry("800x650")

        self.points_data = None
        self.filename_label_var = tk.StringVar(value="Niciun fișier încărcat.")
        self.k_var = tk.StringVar(value="3") # Valoare default pentru K
        self.distance_var = tk.StringVar(value="euclidean") # Valoare default
        self.progress_queue = queue.Queue() # Coadă pentru comunicare cu thread-ul
        self.processing_thread = None # Referință la thread

        # --- Frame pentru control ---
        control_frame = ttk.Frame(self, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Buton încărcare fișier
        load_button = ttk.Button(control_frame, text="Încarcă Puncte (.txt)", command=self.load_file)
        load_button.pack(side=tk.LEFT, padx=5)

        # Etichetă nume fișier
        filename_label = ttk.Label(control_frame, textvariable=self.filename_label_var, wraplength=200)
        filename_label.pack(side=tk.LEFT, padx=5)

        # Input K
        k_label = ttk.Label(control_frame, text="K (clustere):")
        k_label.pack(side=tk.LEFT, padx=(15, 2))
        k_entry = ttk.Entry(control_frame, textvariable=self.k_var, width=5)
        k_entry.pack(side=tk.LEFT, padx=2)

        # Alegere distanță
        distance_label = ttk.Label(control_frame, text="Distanță:")
        distance_label.pack(side=tk.LEFT, padx=(15, 2))
        distance_combo = ttk.Combobox(control_frame, textvariable=self.distance_var, values=["euclidean", "manhattan"], width=10, state="readonly")
        distance_combo.pack(side=tk.LEFT, padx=2)
        distance_combo.set("euclidean") # Setează valoarea default

        # Buton Run
        run_button = ttk.Button(control_frame, text="Rulează K-Medoids", command=self.start_algorithm_thread)
        run_button.pack(side=tk.LEFT, padx=15)
        self.run_button = run_button # Păstrăm referința pentru a-l dezactiva

        # --- Frame pentru grafic ---
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Încorporare Matplotlib
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.ax.set_xlabel("Coordonata X")
        self.ax.set_ylabel("Coordonata Y")
        self.ax.set_title("Așteptare date...")
        self.ax.grid(True)
        self.canvas.draw()
        
        # --- Zonă de log (opțional) ---
        log_frame = ttk.Frame(self, padding="5")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X)
        log_label = ttk.Label(log_frame, text="Log:")
        log_label.pack(anchor=tk.W)
        self.log_text = tk.Text(log_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text['yscrollcommand'] = log_scrollbar.set
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Pornim verificarea cozii
        self.after(100, self.process_queue)


    def log_message(self, message):
        """Adaugă un mesaj în zona de log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END) # Scroll la sfârșit
        self.log_text.config(state=tk.DISABLED)

    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Selectează fișierul cu puncte",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filepath:
            return # Utilizatorul a anulat

        self.points_data = load_points(filepath) # load_points afișează erori dacă e cazul

        if self.points_data is not None:
            self.filename_label_var.set(filepath.split('/')[-1]) # Afișează numele fișierului
            self.log_message(f"Fișier încărcat: {filepath.split('/')[-1]} ({len(self.points_data)} puncte)")
            # Afișează punctele inițiale pe grafic
            self.ax.clear()
            self.ax.scatter(self.points_data[:, 0], self.points_data[:, 1], alpha=0.6, label='Puncte încărcate')
            self.ax.set_xlabel("Coordonata X")
            self.ax.set_ylabel("Coordonata Y")
            self.ax.set_title("Date încărcate")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()
        else:
            self.filename_label_var.set("Eroare la încărcare.")
            self.ax.clear()
            self.ax.set_title("Eroare la încărcarea datelor")
            self.ax.grid(True)
            self.canvas.draw()


    def start_algorithm_thread(self):
        """Pornește algoritmul într-un thread separat."""
        if self.processing_thread and self.processing_thread.is_alive():
             messagebox.showwarning("Ocupat", "Algoritmul K-Medoids rulează deja.")
             return
             
        if self.points_data is None:
            messagebox.showerror("Eroare", "Mai întâi încarcă un fișier cu puncte.")
            return

        try:
            k = int(self.k_var.get())
            if k <= 0:
                raise ValueError("K trebuie să fie un număr pozitiv.")
            if k > len(self.points_data):
                 raise ValueError("K nu poate fi mai mare decât numărul de puncte.")
        except ValueError as e:
            messagebox.showerror("Eroare K", f"Valoare invalidă pentru K: {e}")
            return

        distance_method = self.distance_var.get()
        max_epochs = 100 # Poți face și asta configurabil în GUI
        tolerance = 1e-4

        # Dezactivează butonul Run și curăță log-ul/graficul vechi
        self.run_button.config(state=tk.DISABLED)
        self.log_message("--- Pornire algoritm K-Medoids ---")
        self.ax.clear()
        self.ax.set_title("Procesare K-Medoids...")
        self.ax.grid(True)
        # Desenează punctele din nou ca fundal
        self.ax.scatter(self.points_data[:, 0], self.points_data[:, 1], alpha=0.2, color='gray') 
        self.canvas.draw()
        
        # Golește coada de mesaje vechi
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except queue.Empty:
                break

        # Creează și pornește thread-ul
        self.processing_thread = threading.Thread(
            target=k_medoids_worker,
            args=(self.points_data, k, distance_method, max_epochs, tolerance, self.progress_queue),
            daemon=True # Permite programului să iasă chiar dacă thread-ul rulează
        )
        self.processing_thread.start()


    def process_queue(self):
        """Verifică coada pentru mesaje de la thread și actualizează GUI."""
        try:
            while True: # Procesează toate mesajele disponibile
                 message = self.progress_queue.get_nowait()

                 if message["status"] == "log":
                     self.log_message(message["message"])
                 elif message["status"] == "error":
                     messagebox.showerror("Eroare algoritm", message["message"])
                     self.run_button.config(state=tk.NORMAL) # Reactivează butonul
                 elif message["status"] == "plot_update":
                     self.update_plot(
                        epoch=message["epoch"],
                        medoids=message["medoids"],
                        assignments=message["assignments"],
                        title=message["title"]
                     )
                 elif message["status"] == "finished":
                     self.log_message("--- Algoritm finalizat ---")
                     # Plot final
                     self.update_plot(
                        epoch=message["epoch"],
                        medoids=message["final_medoids"],
                        assignments=message["final_assignments"],
                        title=f"Rezultat Final K-Medoids (Epoca {message['epoch']})"
                     )
                     self.run_button.config(state=tk.NORMAL) # Reactivează butonul
                     final_medoids = message["final_medoids"]
                     self.log_message("\nMedoizi finali:")
                     for i, med in enumerate(final_medoids):
                        self.log_message(f"  Medoid {i}: ({med[0]:.2f}, {med[1]:.2f})")

        except queue.Empty:
             # Nu sunt mesaje noi, continuă să verifici periodic
             pass
        except Exception as e:
            # Eroare neașteptată la procesarea cozii
             print(f"Eroare la procesarea cozii: {e}")
             self.log_message(f"Eroare GUI: {e}")
             if self.processing_thread and self.processing_thread.is_alive():
                 # Încercăm să oprim thread-ul sau doar reactivăm butonul?
                 pass 
             self.run_button.config(state=tk.NORMAL) # Reactivează butonul ca precauție

        # Reprogramează verificarea cozii
        self.after(100, self.process_queue) 

    def update_plot(self, epoch, medoids, assignments, title):
        """Actualizează graficul Matplotlib încorporat."""
        if self.points_data is None: return # Nu avem ce desena
        
        self.ax.clear() # Șterge graficul anterior
        k = len(medoids)
        colors = plt.cm.viridis(np.linspace(0, 1, k)) # Culori consistente

        # Desenează punctele (poate cu alpha mai mic dacă sunt multe)
        # self.ax.scatter(self.points_data[:, 0], self.points_data[:, 1], color='gray', alpha=0.1, label='_nolegend_')
        
        # Desenează punctele clusterizate
        for i in range(k):
             # Verificare suplimentară de siguranță
             if assignments is not None and len(assignments) == len(self.points_data):
                 cluster_points_indices = np.where(assignments == i)[0]
                 if len(cluster_points_indices) > 0:
                     cluster_points = self.points_data[cluster_points_indices]
                     self.ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], alpha=0.6, label=f'Cluster {i+1}' if epoch == 0 or title.startswith("Rezultat Final") else None) # Etichete doar la început/sfârșit
             else:
                 print(f"Avertisment: Asignări invalide pentru plotare în epoca {epoch}, cluster {i}.")
                 
             # Desenează medoidul
             if i < len(medoids): # Verificare suplimentară
                 self.ax.scatter(medoids[i][0], medoids[i][1], color=colors[i], marker='X', s=200, edgecolors='k', label=f'Medoid {i+1}' if epoch == 0 or title.startswith("Rezultat Final") else None)
             else:
                 print(f"Avertisment: Index medoid invalid pentru plotare în epoca {epoch}, index {i}.")


        self.ax.set_title(title)
        self.ax.set_xlabel("Coordonata X")
        self.ax.set_ylabel("Coordonata Y")
        if epoch == 0 or title.startswith("Rezultat Final"):
             self.ax.legend(loc='best', fontsize='small')
        self.ax.grid(True)
        self.canvas.draw() # Re-desenează canvas-ul Tkinter


if __name__ == "__main__":
    app = KMedoidsApp()
    app.mainloop()