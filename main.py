''' 	
	pip install tkinter
	pip install pandas
	pip install pandastable
	pip install numpy
	python -m pip install -U matplotlib
	pip install seaborn
	python -m pip install requests
	pip install -U scikit-learn
	pip install beautifulsoup4
'''
import pandas as pd
from pandastable import Table, TableModel

import numpy as np
import matplotlib as plt
import seaborn as sns
import requests as rq
import time
import math
import datetime
import re

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

from bs4 import BeautifulSoup as bs
from tkinter import *
from tkinter import ttk
import os

# Leemos nuestros dataframes
movies = pd.read_csv("./ml-latest-small/movies.csv", sep=",")
ratings = pd.read_csv("./ml-latest-small/ratings.csv", sep=",")
tags = pd.read_csv("./ml-latest-small/tags.csv", sep=",")
links = pd.read_csv("./ml-latest-small/links.csv", sep=",")

''' Eliminamos nulos
movies.dropna(inplace=True)
ratings.dropna(inplace=True)
tags.dropna(inplace=True)
links.dropna(inplace=True)'''

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
similitud_df = pd.DataFrame(similitud, index=movies['title'], columns=movies['title'])

# Nuestro valor k indicará la cantidad de recomendaciones que damos
def genre_recommendations(i, M, items, k=5):
    '''
    i : str
        Movie (index of the similarity dataframe)
    M : pd.DataFrame
        Similarity dataframe, symmetric, with movies as indices and columns
    items : pd.DataFrame
        Contains both the title and some other features used to define similarity
    k : int
        Amount of recommendations to return
	'''
    index = M.loc[:,i].to_numpy().argpartition(range(-1,-k,-1))
    closest = M.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(i, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)

class Ventana(Frame):
	def __init__(self, master, *args):
		super().__init__( master,*args)

		# Declaración de variables
		self.menu = True
		self.color = True

		self.peli = StringVar()
		self.peli.set("")

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
		Label(self.frame_uno, text="Introduzca el título de una Película:", bg='white', fg= 'black', font= ('Arial', 12, 'bold')).place(relx=0.41, rely=0.15)
		Entry(self.frame_uno, textvariable=self.peli, font=('Arial', 15), highlightthickness=3).place(relx=0.42, rely=0.20)
		Button(self.frame_uno, width=12, text='RECOMENDAR!', bg='red2', fg='white', font= ('Arial', 13, 'bold'), command= lambda : genre_recommendations(self.peli.get(), similitud_df, movies[['title', 'genres']])).place(relx=0.452, rely=0.28)
		self.pt = Table(self.frame_uno, width=760, height=420, dataframe=movies)
		#, showtoolbar=True, showstatusbar=True
		self.pt.show()
		self.pt.place(relx=0.25, rely=0.37)
		
		# Página 2 - 
		Label(self.frame_dos, text='RECOMENDAR PELÍCULAS EN BASE A UN USUARIO', bg='white', fg= 'black', font= ('Arial', 15, 'bold')).place(relx=0.335, rely=0.05)

		# Página 3 - Web Scraping
		Label(self.frame_tres, text='WEB SCRAPING', bg='white', fg= 'black', font= ('Arial', 15, 'bold')).place(relx=0.46, rely=0.05)

		fila_seleccionada = 0
		
		def seleccionPelicula(event):
			rowclicked_single = self.pt.get_row_clicked(event)
			self.pt.setSelectedRow(rowclicked_single)
			self.pt.redraw()
			fila_seleccionada = rowclicked_single
			labTitulo = Label(self.frame_tres, text='Película seleccionada: ' + str(movies.loc[fila_seleccionada, 'title']), bg='white', fg= 'black', font= ('Arial', 15, 'bold')).place(relx=0.3, rely=0.15)

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