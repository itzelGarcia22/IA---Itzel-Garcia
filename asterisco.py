import pygame
import math
from queue import PriorityQueue
import sys

FILAS = 11  
COLS  = 11 
ANCHO_VENTANA = 700
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Algoritmo A* - Visualizador (Corregido)")

BLANCO      = (255, 255, 255)
NEGRO       = (0, 0, 0)
GRIS_CLARO  = (220, 220, 220)
AZUL_OSCURO = (30, 60, 180)      # inicio
VERDE_LIMA  = (50, 205, 50)      # abiertos
ROJO_VIVO   = (220, 20, 60)      # cerrados
AMARILLO    = (255, 215, 0)      # camino
MORADO      = (186, 85, 211)     # fin
NARANJA     = (255, 140, 0)      # paredes
TURQUESA    = (64, 224, 208)     # nodos frontera

# Costes
COST_ARR_ABA = 10
COST_DIAG    = 14

# Fuente
pygame.font.init()
FONT = pygame.font.SysFont('consolas', 12) # Fuente para coordenadas


class Nodo: # Representa cada celda en la cuadrícula
    def __init__(self, fila, col, ancho, total_filas, total_cols):
        self.fila = fila
        self.col = col
        self.x = col * ancho
        self.y = fila * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.total_cols = total_cols
        self.vecinos = []
        self.g = float('inf')
        self.h = float('inf')
        self.f = float('inf')

    def __lt__(self, other): # Para la cola de prioridad
        return self.f < other.f

    def __eq__(self, other): # Para comparar nodos
        return isinstance(other, Nodo) and (self.fila, self.col) == (other.fila, other.col)

    def __hash__(self): # Para usar nodos en conjuntos y diccionarios
        return hash((self.fila, self.col))

    def get_pos(self): # Devuelve la posición del nodo
        return self.fila, self.col

    def es_pared(self): # Verifica si el nodo es una pared
        return self.color == NARANJA

    def es_inicio(self): # Verifica si el nodo es el inicio
        return self.color == AZUL_OSCURO

    def es_fin(self): # Verifica si el nodo es el fin
        return self.color == MORADO

    def restablecer(self): # Restablece el nodo a su estado inicial
        self.color = BLANCO
        self.g = self.h = self.f = float('inf') # Reiniciar costes

    def hacer_inicio(self): self.color = AZUL_OSCURO
    def hacer_fin(self): self.color = MORADO
    def hacer_pared(self): self.color = NARANJA
    def hacer_abierto(self): self.color = TURQUESA
    def hacer_cerrado(self): self.color = ROJO_VIVO
    def hacer_camino(self): self.color = AMARILLO

    def dibujar(self, ventana): # Dibuja el nodo en la ventana
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

        # Mostrar coordenada (fila, col) en la parte superior
        ventana.blit(FONT.render(f"({self.fila},{self.col})", True, NEGRO), (self.x+3, self.y+2))

    def actualizar_vecinos(self, grid): # Actualiza los vecinos del nodo
        self.vecinos = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue
                r = self.fila + dr
                c = self.col + dc
                if 0 <= r < self.total_filas and 0 <= c < self.total_cols:
                    if not grid[r][c].es_pared():
                        self.vecinos.append(grid[r][c])


def heuristica_octile(p1, p2): # Heurística octile
    (r1, c1) = p1
    (r2, c2) = p2
    dx = abs(c1 - c2)
    dy = abs(r1 - r2)
    D = COST_ARR_ABA
    D2 = COST_DIAG
    return D*(dx+dy) + (D2 - 2*D)*min(dx, dy) #donde D=10, D2=14.


def reconstruir_camino(came_from, actual, dibujar): # Reconstruye el camino óptimo
    while actual in came_from:
        actual = came_from[actual]
        if not actual.es_inicio() and not actual.es_fin():
            actual.hacer_camino()
        dibujar()


def algoritmo_a_estrella(dibujar, grid, inicio, fin): # Implementación del algoritmo A*
    for fila in grid:
        for nodo in fila:
            nodo.g = nodo.h = nodo.f = float('inf')

    contador = 0

    open_set = PriorityQueue()
    inicio.g = 0
    inicio.h = heuristica_octile(inicio.get_pos(), fin.get_pos())
    inicio.f = inicio.g + inicio.h

    open_set.put((inicio.f, inicio.h, contador, inicio))
    open_set_hash = {inicio}
    came_from = {}
    lista_abierta = []
    lista_cerrada = []
    camino_optimo = []

    while not open_set.empty(): # Mientras haya nodos por explorar
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Guardar snapshot de la lista abierta
        la_snapshot = [ (nodo.fila, nodo.col) for _,_,_,nodo in open_set.queue if nodo in open_set_hash ]
        lista_abierta.append(list(la_snapshot))

        _, _, _, current = open_set.get()

        if current not in open_set_hash: 
            continue
        open_set_hash.remove(current)
        lista_cerrada.append((current.fila, current.col))

        if not current.es_inicio():
            current.hacer_cerrado()

        if current == fin:
            # Reconstruir camino óptimo
            temp = current
            while temp in came_from:
                camino_optimo.append((temp.fila, temp.col))
                temp = came_from[temp]
            camino_optimo.append((inicio.fila, inicio.col))
            camino_optimo.reverse()
            reconstruir_camino(came_from, fin, dibujar)
            inicio.hacer_inicio()
            fin.hacer_fin()
            # Mostrar en consola
            print("\n--- Lista abierta (por expansión) ---")
            for i, la in enumerate(lista_abierta):
                print(f"Expansión {i+1}: {la}")
            print("\n--- Lista cerrada (orden de expansión) ---")
            print(lista_cerrada)
            print("\n--- Camino óptimo ---")
            print(camino_optimo)
            return True

        for vecino in current.vecinos:
            dr = abs(vecino.fila - current.fila)
            dc = abs(vecino.col - current.col)
            move_cost = COST_DIAG if dr == 1 and dc == 1 else COST_ARR_ABA

            tentative_g = current.g + move_cost
            if tentative_g < vecino.g:
                came_from[vecino] = current
                vecino.g = tentative_g
                vecino.h = heuristica_octile(vecino.get_pos(), fin.get_pos())
                vecino.f = vecino.g + vecino.h

                if vecino not in open_set_hash:
                    contador += 1
                    open_set.put((vecino.f, vecino.h, contador, vecino))
                    open_set_hash.add(vecino)
                    if not vecino.es_fin():
                        vecino.hacer_abierto()

        dibujar()

    print("No se encontró camino.")
    return False


def crear_grid(filas, cols, ancho): # Crea la cuadrícula de nodos
    grid = []
    ancho_nodo = ancho // max(filas, cols)
    for r in range(filas):
        grid.append([])
        for c in range(cols):
            grid[r].append(Nodo(r, c, ancho_nodo, filas, cols))
    return grid


def dibujar_grid(ventana, filas, cols, ancho): # Dibuja la cuadrícula en la ventana
    ancho_nodo = ancho // max(filas, cols)
    for c in range(cols+1):
        pygame.draw.line(ventana, GRIS_CLARO, (c*ancho_nodo, 0), (c*ancho_nodo, ancho_nodo*filas))
    for r in range(filas+1):
        pygame.draw.line(ventana, GRIS_CLARO, (0, r*ancho_nodo), (ancho_nodo*cols, r*ancho_nodo))


def dibujar(ventana, grid, filas, cols, ancho): # Dibuja todo en la ventana
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas, cols, ancho)
    pygame.display.update()


def obtener_click_pos(pos, filas, cols, ancho): # Obtiene la posición del clic en la cuadrícula
    ancho_nodo = ancho // max(filas, cols)
    x, y = pos
    col = x // ancho_nodo
    fila = y // ancho_nodo
    return max(0, min(filas-1, fila)), max(0, min(cols-1, col))


def main(ventana, ancho): # Función principal
    grid = crear_grid(FILAS, COLS, ancho)
    inicio = None
    fin = None

    corriendo = True
    mouse_down_left = False
    mouse_down_right = False
    while corriendo:
        dibujar(ventana, grid, FILAS, COLS, ancho)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if event.type == pygame.MOUSEBUTTONDOWN: # Detectar clics del ratón
                if event.button == 1:
                    mouse_down_left = True
                if event.button == 3:
                    mouse_down_right = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down_left = False
                if event.button == 3:
                    mouse_down_right = False

            if event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN: # Manejar arrastrar y soltar
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, COLS, ancho)
                nodo = grid[fila][col]
                if mouse_down_left:
                    if not inicio and nodo != fin:
                        inicio = nodo
                        inicio.hacer_inicio()
                    elif not fin and nodo != inicio:
                        fin = nodo
                        fin.hacer_fin()
                    elif nodo != fin and nodo != inicio:
                        nodo.hacer_pared()
                if mouse_down_right:
                    nodo.restablecer()
                    if nodo == inicio: inicio = None
                    if nodo == fin: fin = None

            if event.type == pygame.KEYDOWN: # Manejar teclas del teclado
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)
                    algoritmo_a_estrella(lambda: dibujar(ventana, grid, FILAS, COLS, ancho),
                                         grid, inicio, fin)

                if event.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, COLS, ancho)

    pygame.quit()


if __name__ == "__main__": # Instrucciones de uso
    print("\nINSTRUCCIONES DE USO:")
    print("- Click izquierdo: inicio/fin/pared")
    print("- Click derecho: limpiar celda")
    print("- Barra espaciadora: Ejecuta el algoritmo A*.")
    print("- Tecla 'c': Limpia todo el tablero.")
    print("- Cierra la ventana para salir.\n")
    print("COLORES:")
    print("- Azul oscuro: Nodo de inicio")
    print("- Morado: Nodo de fin")
    print("- Naranja: Pared/obstáculo")
    print("- Turquesa: Nodos abiertos (en frontera)")
    print("- Rojo vivo: Nodos cerrados (ya explorados)")
    print("- Amarillo: Camino óptimo encontrado")
    print("- Blanco: Espacio libre\n")
    main(VENTANA, ANCHO_VENTANA)