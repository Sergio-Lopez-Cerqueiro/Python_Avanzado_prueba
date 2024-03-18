import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from skimage.feature import hog
from openpyxl import Workbook

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def cargar_imagenes(
    ruta, tamano
):  # función para realizar la carga del dataset extrayendo la imágen con 2 arrays
    X = []
    y = []
    for clase in os.listdir(ruta):
        carpeta = os.path.join(ruta, clase)
        for archivo in os.listdir(carpeta):
            archivo_path = os.path.join(carpeta, archivo)
            imagen = Image.open(archivo_path)
            imagen = imagen.resize((tamano, tamano))
            imagen = np.array(imagen) / 255.0
            X.append(imagen)
            y.append(clase)
    return np.array(X), np.array(y)


def extraer_caracteristicas_hog(imagen, a, b):
    features, hog_image = hog(
        imagen,
        orientations=9,
        pixels_per_cell=(a, a),
        cells_per_block=(b, b),
        visualize=True,
        channel_axis=2,
    )
    return features


def ejecutar_X_hog(ruta_imagenes, tamano_imagen, pixel, celda):

    # Especifica la ruta de la carpeta que contiene las imágenes
    # ruta_imagenes = (
    #     "D:/Python_clase/VisionArtificial1_23_24/TRABAJO-ENTREGA/archive/train"
    # )
    # # Tamaño al que se redimensionarán las imágenes
    # tamano_imagen = clave
    # Carga las imágenes y etiquetas desde la carpeta (DATASET FINALIZADO)
    print("tamano de imagen: ", tamano_imagen)
    X, y = cargar_imagenes(ruta_imagenes, tamano_imagen)

    # Generate HOG features for each image
    print("Iniciando X_hog:")
    X_hog = [extraer_caracteristicas_hog(imagen, pixel, celda) for imagen in X]
    X_hog = np.array(X_hog)
    print(X_hog)

    return X_hog


if __name__ == "__main__":
    X_hog = []
    X_hog = ejecutar_X_hog(
        "D:/Python_clase/VisionArtificial1_23_24/TRABAJO-ENTREGA/archive/train",
        68,
        8,
        6,
    )
