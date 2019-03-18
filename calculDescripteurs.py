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
dir_images="./images/"
dir_centroid="centers"
#matrice de co-occurence---------------------
def coocurrence(img,niveau):
    	#print("je suis la matrice -----")
		k=cv2.imread(dir_images+img,1)
		k=reduireGray(k,niveau)
		result = skimage.feature.texture.greycomatrix(k, [1,2], [0, np.pi/4, np.pi/2, 3*np.pi/4],normed=True)
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

def create_rep():
    	if not os.path.exists(dir_features):
    			os.makedirs(dir_features)


def trouveCentre():
    	momentsHu=[]
        liste_fichier=[]
        liste_f=[]
    	for file in os.listdir("./"+dir_features):
    		if not os.path.isdir(file):
				fic=open(dir_features+"/"+file,"rb")
				#liste_fichier.append(file)
				#tmp_img=cv2.imread(dir_images+"/"+file,cv2.COLOR_BGR2GRAY)
				#tmp_img=tmp_img.reshape(tmp_img.shape[0]*tmp_img.shape[1])
				#liste_f.append(tmp_img)
				data=pickle.load(fic)
				#print(file,":",data[2])
				momentsHu.append(data[3])
				liste_fichier.append(file)
	return liste_fichier,momentsHu	
create_rep()
input_image=systemCheckAndReturnValue()
rep='./images'
all_files = os.listdir(rep)
directory=os.listdir("./"+dir_features)
for file in all_files:
    #print(file)
	if len(directory)!=0:
    		print("les fichiers existent deja")
		break
	if not os.path.isdir(file):
		histA=histogramme(dir_images, file)
		cok=coocurrence(file,8)
		moment= MomentHU(file)
		nn=file.split(".")
		fichier=nn[0]+".azy"
		nom_fichier=dir_features+"/"+fichier
		data={1:histA,2:moment,3:cok}
		with open(nom_fichier,"ab") as fic:
			pickle.dump(data,fic)

		
		#print("voici son histo:",moment,"\n ",len(moment),"--------------------------------------")
fichiers,m=trouveCentre()
distances=[]
#load_img=cv2.imread('./'+input_image,1)
hu=MomentHU('./'+input_image)
print(m)
kmean=KMeans(n_clusters=10,random_state=0)
kmean.fit(m)
centres=kmean.cluster_centers_
labels=kmean.labels_
print("les centres:",centres)
#print(len(centres))
print(labels)
#print(fichiers)
#print("voici le HU de :",hu)
#retour=kmean.fit_predict(hu)

#print("voici le retour de la fonction predict:",retour)
for mom in centres:
	d=DistanceHu(hu,mom)
	distances.append(d)
max=10000000000
ind=-1
minima=min(distances)
kindice=distances.index(minima)
part=labels[kindice]
print("voici son index:",kindice)
print("voici son groupe:",part)

print("voici le minima:",minima)
for i in range(len(distances)):
    if max>distances[i]:
    	max=distances[i]
        ind=i
print("les distances:",distances)
print("voici l'indice:",ind)
print("il est du groupe:",labels[ind])
print("la liste des fichiers trouV:")
for j in range(len(labels)):
	if labels[j]==labels[ind]:
		print(fichiers[j],":",j)
print("distance:",distances[ind])		
print("")
print("-------------------------------------------ici commence la correspondance-------------------------------------------")
for file in fichiers:
    if not os.path.isdir("./"+dir_features+"/"+file):
    	#print(file)
    	with open("./"+dir_features+"/"+file,"rb") as fic:
			data=pickle.load(fic)
			dist=[]
			for c in centres:
				d=DistanceHu(data[2],c)
				dist.append(d)
			minima=min(dist)
			#print(dist)
			kindice=dist.index(minima)
			groupe=labels[kindice]
			if(groupe==part):
				print("voici son fichier de correspondance:",file,"et son groupe est ",groupe)
    			break	


#while i <nbr_img-1:
    	#print("voici le probleme:",all_files[i])
#		histB=histogramme(rep, all_files[i])
#		img2 = cv2.imread('images/' + all_files[i], 0)
		#fonction de calcul da la distance d'histogramme
#		img2_couleur = cv2.imread('images/' + all_files[i], 1)
 #       matrix_co=coocurrence(img2_couleur)
#		i=i+1
