''' 	
	pip install tkinter
	pip install pandas
	pip install pandastable
	pip install -U scikit-learn
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.metrics.pairwise import cosine_similarity
	from itertools import combinations
	from sklearn.metrics.pairwise import linear_kernel
'''
import pandas as pd
from pandastable import Table

import numpy as np
import matplotlib as plt
import seaborn as sns
import requests as rq
import time
import math
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

from tkinter import *
from tkinter import ttk

# Leemos nuestros dataframes
movies = pd.read_csv("./ml-latest-small/movies.csv", sep=",")
ratings = pd.read_csv("./ml-latest-small/ratings.csv", sep=",")
tags = pd.read_csv("./ml-latest-small/tags.csv", sep=",")
links = pd.read_csv("./ml-latest-small/links.csv", sep=",")
sinopsisdf = pd.read_csv("./ml-latest-small/sinopsisDB.csv", sep=",")

# Eliminamos nuelos
movies.dropna(inplace=True)
ratings.dropna(inplace=True)
tags.dropna(inplace=True)
links.dropna(inplace=True)

# Extraemos el año del título
movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
movies.year = pd.to_datetime(movies.year, format='%Y')
movies.year = movies.year.dt.year
movies.title = movies.title.str[:-7]

# TF-IDF para las diferentes combinaciones de géneros
tf = TfidfVectorizer(analyzer=lambda s: (c for i in range(1,4)
                     for c in combinations(s.split('|'), r=i)))

# Formamos nuestra matriz con los géneros
matriz = tf.fit_transform(movies.genres)
# Aplicamos a esta matriz la similitud del coseno
similitud = cosine_similarity(matriz)
# Creamos un DF con esta similitud
similitudG= pd.DataFrame(similitud, index=movies['title'], columns=movies['title'])

class Ventana(Frame):
	def __init__(self, master, *args):
		super().__init__( master,*args)

		# Declaración de variables
		self.menu = True
		self.color = True

		# Settings frame principal
		self.frame_top = Frame(self.master, bg='white', height = 50)
		self.frame_top.grid(column = 1, row = 0, sticky='nsew')
		self.frame_principal = Frame(self.master, bg='white')
		self.frame_principal.grid(column=1, row=1, sticky='nsew')
		self.master.columnconfigure(1, weight=1)
		self.master.rowconfigure(1, weight=1)
		self.frame_principal.columnconfigure(0, weight=1)
		self.frame_principal.rowconfigure(0, weight=1)
		self.nav()		

	def pantalla_inicio(self):
		self.paginas.select([self.frame_principal])
		self.frame_uno.columnconfigure(0, weight=1)
		self.frame_uno.columnconfigure(1, weight=1)
		
	def pantalla_pelicula(self):
		self.paginas.select([self.frame_uno])
		self.frame_uno.columnconfigure(0, weight=1)
		self.frame_uno.columnconfigure(1, weight=1)

	def pantalla_usuario(self):
		self.paginas.select([self.frame_dos])
		self.frame_dos.columnconfigure(0, weight=1)
		self.frame_dos.columnconfigure(1, weight=1)

	def pantalla_webscraping(self):
		self.paginas.select([self.frame_tres])
		self.frame_tres.columnconfigure(0, weight=1)
		self.frame_tres.columnconfigure(1, weight=1)

	def nav(self):
		self.imagen_pelicula = PhotoImage(file ='./img/ranking.png')
		self.imagen_usuario = PhotoImage(file ='./img/user.png')
		self.imagen_webscraping = PhotoImage(file ='./img/movie.png')
		self.imagen_home = PhotoImage(file ='./img/home.png')
		self.logo = PhotoImage(file ='./img/logo.png')

		self.paginas = ttk.Notebook(self.frame_principal, style= 'TNotebook')
		self.paginas.grid(column=0, row=0, sticky='nsew')
		self.frame_inicio = Frame(self.paginas, bg='#ffbe57')
		self.frame_uno = Frame(self.paginas, bg='white')
		self.frame_dos = Frame(self.paginas, bg='white')
		self.frame_tres = Frame(self.paginas, bg='white')
		self.paginas.add(self.frame_inicio, image = self.imagen_home)
		self.paginas.add(self.frame_uno, image = self.imagen_pelicula)
		self.paginas.add(self.frame_dos, image = self.imagen_usuario)
		self.paginas.add(self.frame_tres, image = self.imagen_webscraping)

		# Inicio
		cl = Button(self.frame_top, text="Exit",width=5, height=2, bg='red2', fg='white', font= ('Arial', 8, 'bold'), command=self.quit)
		cl.place(relx=0.965, rely=0.10)
		Label(self.frame_top, text='Bienvenido a nuestro Proyecto Final de SSII', bg='white', fg='black', font=('Arial', 25, 'bold')).place(relx=0.28, rely=0.10)
		Label(self.frame_inicio, image= self.logo, bg='white').pack(expand=1, pady=0)
		Label(self.frame_inicio, text= 'Para empezar, seleccione un icono del menú.', bg='#ffbe57', fg= 'white', font= ('Arial', 20, 'bold')).pack(expand=1)

		# Página 1 - Recomendador en base a una película
		Label(self.frame_uno, width=60, text='RECOMENDAR PELÍCULAS EN BASE A UNA PELÍCULA', bg='white', fg= 'black', font= ('Arial', 15, 'bold')).place(relx=0.26, rely=0.05)
		Label(self.frame_uno, text="Introduzca el título de una película:", bg='white', fg= 'black', font= ('Arial', 12, 'bold')).place(relx=0.41, rely=0.15)
		peli = 'none'
		entryPeli = Entry(self.frame_uno, textvariable=peli, font=('Arial', 15), highlightthickness=3)
		entryPeli.place(relx=0.42, rely=0.20)
	
		def generarRecomendacionesGenero(i, M, items, k=10):
			'''
			i - nombre de la película
			M - dataframe de similitud
			items - dataframe de títulos y géneros
			k - número de recomendaciones
			'''

			if (i != ''):
				ventana_secundaria = Toplevel()
				ventana_secundaria.title("Recomendación por género")
				ventana_secundaria.config(width=500, height=1000)
				Label(ventana_secundaria, text='Recomendaciones para: ' + str(i), bg='white', fg= 'black', font= ('Arial', 10, 'bold')).place(relx=0.3, rely=0.05)

				index = M.loc[:,i].to_numpy().argpartition(range(-1,-k,-1))
				closest = M.columns[index[-1:-(k+2):-1]]
				closest = closest.drop(i, errors='ignore')
				recomendaciones = pd.DataFrame(closest).merge(items).head(k)
				print(recomendaciones)
				self.pt = Table(ventana_secundaria, width=400, height=600, dataframe=recomendaciones)
				self.pt.show()
				self.pt.place(relx=0.25, rely=0.37)
			else:
				print('Película no encontrada')
			
		def generarRecomendacionesSinopsis(i, M, items, k=10):
			ventana_secundaria = Toplevel()
			ventana_secundaria.title("Recomendación por sinopsis")
			ventana_secundaria.config(width=500, height=1000)
			Label(ventana_secundaria, text='Recomendaciones para: ' + str(i), bg='white', fg= 'black', font= ('Arial', 10, 'bold')).place(relx=0.3, rely=0.05)
			index = M.loc[:,i].to_numpy().argpartition(range(-1,-k,-1))
			closest = M.columns[index[-1:-(k+2):-1]]
			closest = closest.drop(i, errors='ignore')
			recomendaciones = pd.DataFrame(closest).merge(items).head(k)
			self.pt = Table(ventana_secundaria, width=280, height=280, dataframe=recomendaciones)
			self.pt.show()
			self.pt.place(relx=0.25, rely=0.37)

		Button(self.frame_uno, width=26, text='RECOMENDAR POR GÉNEROS!', bg='red2', fg='white', font= ('Arial', 13, 'bold'), command= lambda : generarRecomendacionesGenero(entryPeli.get(), similitudG, movies[['title', 'genres']])).place(relx=0.4, rely=0.28) #
		Button(self.frame_uno, width=26, text='RECOMENDAR POR SINOPSIS!', bg='red2', fg='white', font= ('Arial', 13, 'bold'), command= lambda : generarRecomendacionesSinopsis(entryPeli.get(), similitudG, movies['title', 'genres'])).place(relx=0.4, rely=0.34)
		self.pt = Table(self.frame_uno, width=760, height=420, dataframe=movies)
		#, showtoolbar=True, showstatusbar=True
		self.pt.show()
		self.pt.place(relx=0.25, rely=0.44)
		
		# Página 2 - Recomendador en base a un usuario
		Label(self.frame_dos, text='RECOMENDAR PELÍCULAS EN BASE A UN USUARIO', bg='white', fg= 'black', font= ('Arial', 15, 'bold')).place(relx=0.335, rely=0.05)

		# Página 3 - Web Scraping
		Label(self.frame_tres, text='WEB SCRAPING', bg='white', fg= 'black', font= ('Arial', 15, 'bold')).place(relx=0.46, rely=0.05)
		
		fila_seleccionada = 0
		def seleccionPelicula(event):
			rowclicked_single = self.pt.get_row_clicked(event)
			self.pt.setSelectedRow(rowclicked_single)
			self.pt.redraw()
			fila_seleccionada = rowclicked_single
			labTitulo = Label(self.frame_tres, text='Película seleccionada: ' + str(movies.loc[fila_seleccionada, 'title']), bg='white', fg= 'black', font= ('Arial', 15, 'bold')).place(relx=0.35, rely=0.15)
			movie_id_sinopsis = int(movies.loc[fila_seleccionada, 'movieId'])
			print(movie_id_sinopsis)
			labSinopsis = Label(self.frame_tres, text='Sinopsis: ' + str(sinopsisdf.loc[movie_id_sinopsis, 'sinopsis']), bg='white', fg= 'black', font= ('Arial', 15, 'bold')).place(relx=0.35, rely=0.35)
			labRating = Label(self.frame_tres, text='Rating: ' + str(sinopsisdf.loc[movie_id_sinopsis, 'rating']), bg='white', fg= 'black', font= ('Arial', 15, 'bold')).place(relx=0.35, rely=0.55)
	
		fila = self.pt.rowheader.bind('<Button-1>', seleccionPelicula)

# Settings ventana
if __name__ == "__main__":
    ventana = Tk()
    ventana.title("SSII - MovieLens")
    ventana.resizable(True, True)
    ventana.attributes('-fullscreen',True)
    ventana.call('wm', 'iconphoto', ventana._w, PhotoImage(file='./img/logo.png'))

    app = Ventana(ventana)
    ventana.mainloop()