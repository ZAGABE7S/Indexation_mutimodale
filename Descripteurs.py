###UTILISATION##################################
#
# python Descripteurs.py 'nom_image_requette'
###############################################""


## importation des packages
import sys
import pickle
from scipy.spatial import distance as dist
import numpy as np
np.seterr(over='ignore')
import argparse
import glob
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import os
from math import sqrt
import csv
import random
import math
import operator
import skimage.feature

nbr_class=72
voisin=5
w=0.33
dir_features="features";
dir_images="./images/";
#matrice de co-occurence---------------------
def coocurrence(image):
    	#print("je suis la matrice -----")
        result = skimage.feature.texture.greycomatrix(image, [1,2], [0, np.pi/4, np.pi/2, 3*np.pi/4],normed=True)
        return result
def energie(cooccurence1):
    	res=0
        for m in cooccurence1:
            for n in m:
    				for i in range(n.shape[0]):
    						for j in range(n.shape[1]):
    								res=res+(n[i,j])**2
        return res
def inertie(cooccurence1):
    	res=0
        for m in cooccurence1:
            for n in m:
    				for i in range(n.shape[0]):
    						for j in range(n.shape[1]):
    								res=res+((i-j)**2*n[i,j])
        return res		
def entropie(cooccurence1):
    	res=0
        for m in cooccurence1:
            for n in m:
    				for i in range(n.shape[0]):
    						for j in range(n.shape[1]):
    								tmp=math.log(n[i,j])if n[i,j]>0 else 0
    								res=res+(n[i,j]*tmp)
        return res	
def momentDif(cooccurence1):
    	res=0
        for m in cooccurence1:
            for n in m:
    				for i in range(n.shape[0]):
    						for j in range(n.shape[1]):
    								res=res+((1/(1+(i-j)**2))*n[i,j])
        return res			

#fonction de distance entre les texture
def distTexture(im,im1):
    	en=(energie(im)-energie(im1))**2
        inert=(inertie(im)-inertie(im1))**2
        entrop=(entropie(im1)-entropie(im))**2
        mom=(momentDif(im)-momentDif(im1))**2
        g=en+inert+entrop+mom
        return math.sqrt(g)/4

def reduireGray(image_gray,T_value):
    	image_gray=cv2.cvtColor(image_gray,cv2.COLOR_BGR2GRAY)
    	for i in range(image_gray.shape[0]):
            for j in range(image_gray.shape[1]):
    			image_gray[i,j]=image_gray[i,j]//T_value
        return image_gray		
#une fonction pour verifier les donnees en entree-----
def Fmeasure(prec,rec,alpha):
    f=0
    if (prec!=0.0 and rec!=0.0):
    		f=1/((alpha*1/prec)+(1-alpha)*(1/rec))
    return float(f) 

#ici on trouve la precision--------------------	
def precision(relevant_retreived,retreived_data):	
	return float(float(relevant_retreived)/float(retreived_data));

#retourne le nom de classe selon la logique du programme
def getNomClasse(nom):
    	classe=nom[3:6]
        if(classe[1]=="_"):
    			return classe[0]
        if(classe[2]=="_"):
    			return classe[0:2]	
        return classe	

#retourne le Recall--------------------------------
def recall_data(relevant_retreived,elInClass):
    return float(relevant_retreived)/float(elInClass)

#retourne le nombre d'elements de la meme classe que l'image en requete
def getNumberElmentOfSameClassThan(el,liste):
    nb=0
    classe=getNomClasse(el)
    #print(len(liste))
    ind=0
    for u in liste:
    		#print(u)
		if classe==getNomClasse(u):
    				nb=nb+1
    return nb
#----------------------------------------------------------------------------- 	
def systemCheckAndReturnValue():
    print(sys.argv)
    l=len(sys.argv)
    if(l<3):
    	print("nombre d'argument est insuffissant")	
	sys.exit(1);
    if(sys.argv[1]!="search"):
	print("il ya pas le mot cles search dans votre requete",sys.argv[1])
	sys.exit(1)
    return sys.argv[2]		
	
def azyTri(liste,liste1):
    for i in range(len(liste)):
    		for j in range(len(liste)):
			if(liste[i]<liste[j]):
				tmp=liste[i]
				liste[i]=liste[j]
				liste[j]=tmp
				tmp1=liste1[i]
				liste1[i]=liste1[j]
				liste1[j]=tmp1
    return [liste,liste1]				
	#fonction de calcul d'histogramme 
def histogramme(rep, img):
	
	nb_bits=32
	
	x=1
	hist=[]
	#lecture des images de la base  
	img = cv2.imread(rep + img)
	#split de la base de couleur RGB
	#print("elle est la:",img)
	b, g, r = cv2.split(img)
	# histogramme de B 
	histrb = cv2.calcHist([img],[0],None,[nb_bits],[0,nb_bits])
	#histogramme de G 
	histrg = cv2.calcHist([img],[1],None,[nb_bits],[0,nb_bits])
	#histogramme R
	histrr = cv2.calcHist([img],[2],None,[nb_bits],[0,nb_bits])
	#norlise les histograme
	histrb=cv2.normalize(histrb,histrb)
	histrg=cv2.normalize(histrg,histrg)
	histrr=cv2.normalize(histrr,histrr)

	#sommation des histogramme de RGB 
	#for i in range(len(histrb)):
		#hist.append( float(histrb[i]))
	#for i in range(len(histrg)):
		#hist.append( float(histrg[i]))
	#for i in range(len(histrr)):
		#hist.append( float(histrr[i]))
	hist=histrb+histrg+histrr	
	return hist # returne la valeur de l'histogramme des trois couleurs 
# Calcul de la distancee entre deux images de la base 
def distance(histA, histB):
	histg=0
	sommaj=0
	d_global=[] # liste des distances 
	#parcour des valeurs des histogrammes A t B
	for i in range(len(histA)):

		#histg = abs(float(histA[i])- float(histB[i]))
		histg = histg+abs(float(histA[i])- float(histB[i]))
		sommaj=sommaj+histA[i]
		#d_global.append( histg/sum(histA))# insere toutes las valeurs dans la liste 
	d_global.append( histg/sommaj)	
		
	return	 float(sum(d_global)) # retourne la distance entre les deux histogrammes

##Implementation manuelle du moment de Hu "reference wikippedia"



### Calcul du moment du Hu en utilisant la fonction de opencv	
def MomentHU (image):
	#calul du moment de l'image 
	img = cv2.imread(image,0)
	HU=cv2.HuMoments(cv2.moments(img)).flatten()
	return HU # return le moment de hu de l'image 
	
	
def DistanceHu(moment1,moment2):
	dist=0
	#moment1= MomentHU(im1)# moment de l'image1
	#moment2= MomentHU(im2)# moment de l'image2
	#parcour des valeurs du moment
	for i in range (len(moment1)):
		#calcul de la distance
		dist= dist +((moment1[i]-moment2[i])**2)
	DisFinal= np.sqrt(dist)/len(moment1)
	return   DisFinal # retourne la distance de HU 
#Calcul de la distance globale
def weigthHuandColor(distHist,distHu,distT):
    return (distHist*w)+(w*distHu)+(w*distT)

def loaddFeature(image_ent,value):
    	
	poids=[]
	for file in os.listdir("./"+dir_features):
		if not os.path.isdir(file):
			fic=open(dir_features+"/"+file,"rb")	
			data=pickle.load(fic)
			dist_texture=distTexture(image_ent[3],data[3])
			dist_hist=distance(image_ent[1],data[1])
			dist_mo=DistanceHu(image_ent[2],data[2])
			poids.append(weigthHuandColor(dist_hist,dist_mo,dist_texture))
			print("sommation des distances :",weigthHuandColor(dist_hist,dist_mo,dist_texture))

	k_voisin=[]
	poids,all_files=azyTri(poids,os.listdir("./"+dir_features))
	for i in range(voisin):
    		print(all_files[i],"=",poids[i])
	for i in range (voisin):
		k_voisin.append(all_files[i])
		print("---------------------") 
	print("voici les k voisin:",k_voisin)	
	nombre=getNumberElmentOfSameClassThan(value,k_voisin)
	prec=precision(nombre,len(k_voisin))
	rec=recall_data(prec,nbr_class)
	print(prec," et ",rec)
	f=Fmeasure(prec,rec,0.5)
	print("voici le nombre:",nombre," et la precision:",prec," nbre element voisin ",len(k_voisin))
	print("le recall:",rec," et la mesure F est: ",f)
    			

def main():
	# variables 
	prop=[]
	value=systemCheckAndReturnValue()
	#print("voici le value:",value)
	img1 = cv2.imread('./'+value,0)
	distanceTexture=[]
	k=cv2.imread('./'+value,1)
	k=reduireGray(k,8)
	cok=coocurrence(k)
	rep='./images/'
	all_files = os.listdir(rep)
	#print("fichier:",all_files);
	nbr_img=len(all_files)
	i=1
	histA=histogramme("./", value)
	moment=MomentHU("./"+value)
	#on recupere l'entite de l'image par 
	#son histogramme
	#son moment de Hu
	# et par sa matrix de Cooccurence---------------------
	image_ent={1:histA,2:moment,3:cok}
	#print(image_ent)
	loaddFeature(image_ent,value)




	histB=[];d1=[];d2=[];poids = []
	#print "Distance globale avec histogrammme a 8 bits"
	#print "Calcul de distances avec les moments de HU"
	print ("Calcul de la distance globale(histogramme 8 bits + moment de hu)")
	#print("le moment hu:",MomentHU(img1))
	distanceFichier=[]

	
		
if __name__ == "__main__":
		main()
