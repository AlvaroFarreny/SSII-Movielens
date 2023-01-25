''' 	
	pip install tkinter
	pip install pandas
	pip install pandastable
	pip install -U scikit-learn
'''
import pandas as pd
from pandastable import Table

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

from tkinter import *
from tkinter import ttk

import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt


# Leemos nuestros dataframes
movies = pd.read_csv("./ml-latest-small/movies.csv", sep=",")
movies_sinopsis = pd.read_csv("./movies_sinopsis.csv", sep=",")
ratings_df = pd.read_csv("./ml-latest-small/ratings.csv", sep=",")
tags = pd.read_csv("./ml-latest-small/tags.csv", sep=",")
links = pd.read_csv("./ml-latest-small/links.csv", sep=",")
#similitudG = pd.read_csv("./ml-latest-small/similitudG.csv", sep=",")
#similitudS1 = pd.read_csv("./ml-latest-small/similitudS1.csv", sep=",")
#similitudS2 = pd.read_csv("./ml-latest-small/similitudS2.csv", sep=",")
#similitudS3 = pd.read_csv("./ml-latest-small/similitudS3.csv", sep=",")
#similitudS4 = pd.read_csv("./ml-latest-small/similitudS4.csv", sep=",")
#similitudS5 = pd.read_csv("./ml-latest-small/similitudS5.csv", sep=",")
#sinopsisdf = pd.read_csv("./ml-latest-small/sinopsisDB.csv", sep=",")

# Extraemos el año del título
movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
movies.year = pd.to_datetime(movies.year, format='%Y')
movies.year = movies.year.dt.year
movies.title = movies.title.str[:-7]


class Ventana(Frame):
    def __init__(self, master, *args):
        super().__init__(master, *args)

        # Declaración de variables
        self.menu = True
        self.color = True

        self.peli = StringVar()
        self.peli.set("")
        # Página 2
        self.id_usuario = StringVar()

        # Settings frame principal
        self.frame_top = Frame(self.master, bg='white', height=50)
        self.frame_top.grid(column=1, row=0, sticky='nsew')
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
        self.imagen_pelicula = PhotoImage(file='./img/ranking.png')
        self.imagen_usuario = PhotoImage(file='./img/user.png')
        self.imagen_webscraping = PhotoImage(file='./img/movie.png')
        self.imagen_home = PhotoImage(file='./img/home.png')
        self.logo = PhotoImage(file='./img/logo.png')

        self.paginas = ttk.Notebook(self.frame_principal, style='TNotebook')
        self.paginas.grid(column=0, row=0, sticky='nsew')
        self.frame_inicio = Frame(self.paginas, bg='#ffbe57')
        self.frame_uno = Frame(self.paginas, bg='white')
        self.frame_dos = Frame(self.paginas, bg='white')
        self.frame_tres = Frame(self.paginas, bg='white')
        self.paginas.add(self.frame_inicio, image=self.imagen_home)
        self.paginas.add(self.frame_uno, image=self.imagen_pelicula)
        self.paginas.add(self.frame_dos, image=self.imagen_usuario)
        self.paginas.add(self.frame_tres, image=self.imagen_webscraping)

        # Inicio
        cl = Button(self.frame_top, text="Exit", width=5, height=2, bg='red2',
                    fg='white', font=('Arial', 8, 'bold'), command=self.quit)
        cl.place(relx=0.965, rely=0.10)
        Label(self.frame_top, text='Bienvenido a nuestro Proyecto Final de SSII',
              bg='white', fg='black', font=('Arial', 25, 'bold')).place(relx=0.28, rely=0.10)
        Label(self.frame_inicio, image=self.logo,
              bg='white').pack(expand=1, pady=0)
        Label(self.frame_inicio, text='Para empezar, seleccione un icono del menú.',
              bg='#ffbe57', fg='white', font=('Arial', 20, 'bold')).pack(expand=1)

        # Página 1 - Recomendador en base a una película
        Label(self.frame_uno, width=60, text='RECOMENDAR PELÍCULAS EN BASE A UNA PELÍCULA',
              bg='white', fg='black', font=('Arial', 15, 'bold')).place(relx=0.26, rely=0.05)
        Label(self.frame_uno, text="Introduzca el título de una Película:", bg='white',
              fg='black', font=('Arial', 12, 'bold')).place(relx=0.41, rely=0.15)
        ent1 = Entry(self.frame_uno, textvariable=self.peli, font=(
            'Arial', 15), highlightthickness=3).place(relx=0.42, rely=0.20)

        def generarRecomendacionesGenero(i, M, items, k=10):
            '''
            i - nombre de la película
            M - dataframe de similitud
            items - dataframe de títulos y géneros
            k - número de recomendaciones
            '''
            ventana_secundaria = Toplevel()
            ventana_secundaria.title("Recomendación por género")
            ventana_secundaria.config(width=300, height=200)
            Label(ventana_secundaria, text='Recomendaciones para: ' + str(i), bg='white',
                  fg='black', font=('Arial', 10, 'bold')).place(relx=0.3, rely=0.05)
            index = M.loc[:, i].to_numpy().argpartition(range(-1, -k, -1))
            closest = M.columns[index[-1:-(k+2):-1]]
            closest = closest.drop(i, errors='ignore')
            recomendaciones = pd.DataFrame(closest).merge(items).head(k)
            self.pt = Table(ventana_secundaria, width=280,
                            height=280, dataframe=recomendaciones)
            self.pt.show()
            self.pt.place(relx=0.25, rely=0.37)

        def generarRecomendacionesSinopsis(i, M, items, k=10):
            ventana_secundaria = Toplevel()
            ventana_secundaria.title("Recomendación por sinopsis")
            ventana_secundaria.config(width=300, height=200)
            Label(ventana_secundaria, text='Recomendaciones para: ' + str(i), bg='white',
                  fg='black', font=('Arial', 10, 'bold')).place(relx=0.3, rely=0.05)
            index = M.loc[:, i].to_numpy().argpartition(range(-1, -k, -1))
            closest = M.columns[index[-1:-(k+2):-1]]
            closest = closest.drop(i, errors='ignore')
            recomendaciones = pd.DataFrame(closest).merge(items).head(k)
            self.pt = Table(ventana_secundaria, width=280,
                            height=280, dataframe=recomendaciones)
            self.pt.show()
            self.pt.place(relx=0.25, rely=0.37)

        Button(self.frame_uno, width=26, text='RECOMENDAR POR GÉNEROS!', bg='red2', fg='white', font=('Arial', 13, 'bold'),
               command=lambda: generarRecomendacionesGenero('Toy Story', similitud_dfG, movies[['title', 'genres']])).place(relx=0.4, rely=0.28)
        Button(self.frame_uno, width=26, text='RECOMENDAR POR SINOPSIS!', bg='red2', fg='white', font=('Arial', 13, 'bold'),
               command=lambda: generarRecomendacionesSinopsis('Toy Story', similitud_dfS, movies[['title', 'genres']])).place(relx=0.4, rely=0.34)
        self.pt = Table(self.frame_uno, width=760,
                        height=420, dataframe=movies)
        # , showtoolbar=True, showstatusbar=True
        self.pt.show()
        self.pt.place(relx=0.25, rely=0.44)

        # Página 2 - Recomendador en base a un usuario
        Label(self.frame_dos, text='RECOMENDAR PELÍCULAS EN BASE A UN USUARIO',
              bg='white', fg='black', font=('Arial', 15, 'bold')).place(relx=0.335, rely=0.05)
        Label(self.frame_dos, text="Introduzca el ID de un usuario:", bg='white',
              fg='black', font=('Arial', 12, 'bold')).place(relx=0.41, rely=0.15)
        ent2 = Entry(self.frame_dos, textvariable=self.id_usuario, font=(
            'Arial', 15), highlightthickness=3).place(relx=0.42, rely=0.20)
        Button(self.frame_dos, width=26, text='RECOMENDAR POR USUARIOS!', bg='red2', fg='white', font=('Arial', 13, 'bold'),
               command=lambda: recomendar_a_usuario(self.id_usuario)).place(relx=0.4, rely=0.28)

        def recomendar_a_usuario(user_id):
            user_id = int(self.id_usuario.get())
            print(user_id)
            ratings = ratings_df
            # No queremos trabajar con timestamp
            user_ids = ratings["userId"].unique().tolist()
            user2user_encoded = {x: i for i, x in enumerate(user_ids)}
            userencoded2user = {i: x for i, x in enumerate(user_ids)}
            movie_ids = ratings["movieId"].unique().tolist()
            movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
            movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
            ratings["user"] = ratings["userId"].map(user2user_encoded)
            ratings["movie"] = ratings["movieId"].map(movie2movie_encoded)

            num_users = len(user2user_encoded)
            num_movies = len(movie_encoded2movie)
            ratings["rating"] = ratings["rating"].values.astype(np.float32)
            # min and max ratings will be used to normalize the ratings later
            min_rating = min(ratings["rating"])
            max_rating = max(ratings["rating"])

            ratings = ratings.sample(frac=1, random_state=42)
            x = ratings[["user", "movie"]].values
            # Normalize the targets between 0 and 1. Makes it easy to train.
            y = ratings["rating"].apply(lambda x: (
                x - min_rating) / (max_rating - min_rating)).values
            # Assuming training on 90% of the data and validating on 10%.
            train_indices = int(0.9 * ratings.shape[0])
            x_train, x_val, y_train, y_val = (
                x[:train_indices],
                x[train_indices:],
                y[:train_indices],
                y[train_indices:],
            )
            EMBEDDING_SIZE = 50
            model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
            model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
            )
            # Let us get a user and see the top recommendations.
            movies_watched_by_user = ratings[ratings.userId == user_id]
            movies_not_watched = movies[
                ~movies["movieId"].isin(movies_watched_by_user.movieId.values)
            ]["movieId"]
            movies_not_watched = list(
                set(movies_not_watched).intersection(
                    set(movie2movie_encoded.keys()))
            )
            movies_not_watched = [
                [movie2movie_encoded.get(x)] for x in movies_not_watched]
            user_encoder = user2user_encoded.get(user_id)
            user_movie_array = np.hstack(
                ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
            )
            ratings = model.predict(user_movie_array).flatten()
            top_ratings_indices = ratings.argsort()[-10:][::-1]
            recommended_movie_ids = [
                movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
            ]
            Label(self.frame_dos, text="Showing recommendations for user: {}".format(user_id),
                  bg='white', fg='black', font=('Arial', 11, 'bold')).place(relx=0.42, rely=0.35)

            print("Showing recommendations for user: {}".format(user_id))
            print("====" * 9)
            print("Movies with high ratings from user")
            print("----" * 8)
            top_movies_user = (
                movies_watched_by_user.sort_values(
                    by="rating", ascending=False)
                .head(5)
                .movieId.values
            )
            movie_df_rows = movies[movies["movieId"].isin(top_movies_user)]

            lista_pelis = []
            for row in movie_df_rows.itertuples():
                #De aqui queriamos sacarlo para mostrarlo en labels
                lista_pelis.append(row.title)
                print(row.title, ":", row.genres)

            print("----" * 8)
            print("Top 10 movie recommendations")
            print("----" * 8)
            recommended_movies = movies[movies["movieId"].isin(
                recommended_movie_ids)]
            for row in recommended_movies.itertuples():
                print(row.title, ":", row.genres)
        # Página 3 - Web Scraping
        Label(self.frame_tres, text='WEB SCRAPING', bg='white', fg='black',
              font=('Arial', 15, 'bold')).place(relx=0.46, rely=0.05)

        fila_seleccionada = 0

        def seleccionPelicula(event):
            rowclicked_single = self.pt.get_row_clicked(event)
            self.pt.setSelectedRow(rowclicked_single)
            self.pt.redraw()
            fila_seleccionada = rowclicked_single
            labTitulo = Label(self.frame_tres, text='Película seleccionada: ' + str(
                movies_sinopsis.loc[fila_seleccionada, 'title']), bg='white', fg='black', font=('Arial', 15, 'bold')).place(relx=0.35, rely=0.15)
            labSinopsis = Label(self.frame_tres, text='Sinopsis: ' + str(
                movies_sinopsis.loc[fila_seleccionada, 'sinopsis']), bg='white', fg='black', font=('Arial', 15, 'bold')).place(relx=0.35, rely=0.35)
            labRating = Label(self.frame_tres, text='Rating: ' + str(
                movies_sinopsis.loc[fila_seleccionada, 'rating']), bg='white', fg='black', font=('Arial', 15, 'bold')).place(relx=0.35, rely=0.55)

        fila = self.pt.rowheader.bind('<Button-1>', seleccionPelicula)


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)


# Settings ventana
if __name__ == "__main__":
    ventana = Tk()
    ventana.title("SSII - MovieLens")
    ventana.resizable(True, True)
    ventana.attributes('-fullscreen', True)
    ventana.call('wm', 'iconphoto', ventana._w,
                 PhotoImage(file='./img/logo.png'))

    app = Ventana(ventana)
    ventana.mainloop()
