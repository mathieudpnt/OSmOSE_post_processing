# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:02:46 2017

@author: richard Dréo
"""


import numpy as np
from operator import isub
import math

class TrajectoryFda:
    
    # constructeur de la classe
    def __init__(self, mmsi, epsilon=0.001, nbDepassementTolere=3):
        
        self.mmsi = mmsi
        self.timestamp=np.empty([1,0])
        self.latitude=np.empty([1,0])
        self.longitude=np.empty([1,0])
        self.degree=1
        
        # variables pour la latitude
        self.latitudeResultArray=np.zeros((0,7))
        
        # variables pour la longitude
        self.longitudeResultArray=np.zeros((0,7))
        
        # tolerance spatiale
        self.epsilon=epsilon
                
    def __version__(self):
        return '0.0.1'
    
    # fonction a appeler pour ajouter des donnnées AIS
    def setNewData(self,timestamp,latitude, longitude):      
    
    
        if timestamp==self.timestamp[-1]:
            return  
        # on refuse les doublons dus aux messages "repeat>0"  
        if latitude==self.latitude[-1] and longitude==self.longitude[-1]:
            return
         
        if self.timestamp.size>0:
            distanceNq=self.coordinates2DistanceInMeters(latitude, longitude, self.latitude[-1],self.longitude[-1])/1852.0
            timeH=(timestamp-self.timestamp[-1])/3600.0
            speedConsistency= distanceNq/timeH
            
            if speedConsistency>50:
                print('mmsi',self.mmsi, 'speed:', speedConsistency, 'ts:', timestamp)
                return
                
        # on stocke les données
        self.timestamp=np.append(self.timestamp,timestamp)
        self.latitude=np.append(self.latitude,latitude)
        self.longitude=np.append(self.longitude,longitude)
        
        if self.timestamp.size<2:
            return
            
        # on lance le processus
        self.processTrajectoryLongitude()
        self.processTrajectoryLatitude()
        return      
        
    def getLatitudeFda(self):
        return self.latitudeResultArray
        
    def getLongitudeFda(self):
        return self.longitudeResultArray
        
    def getPosition(self, timestamp):
        currentLatFda=np.logical_and(self.latitudeResultArray[:,0]<=timestamp,self.latitudeResultArray[:,1]>timestamp)
        currentLonFda=np.logical_and(self.longitudeResultArray[:,0]<=timestamp,self.longitudeResultArray[:,1]>timestamp)
        
        ii_lat=np.argwhere(currentLatFda)
        ii_lon=np.argwhere(currentLonFda)
        
        
        lat_t=np.polyval([self.latitudeResultArray[ii_lat,2],self.latitudeResultArray[ii_lat,3],self.latitudeResultArray[ii_lat,4],
                          self.latitudeResultArray[ii_lat,5]],timestamp)
        lon_t=np.polyval([self.longitudeResultArray[ii_lon,2],self.longitudeResultArray[ii_lon,3],self.longitudeResultArray[ii_lon,4],
                          self.longitudeResultArray[ii_lon,5]],timestamp)                          
        return lat_t,lon_t
        
     
    def addNewLatitudeFunction(self,startTime, endTime, f, nbpointsCompressed):
        self.latitudeResultArray=np.vstack((self.latitudeResultArray, [startTime,endTime,f[0],f[1],f[2], f[3], nbpointsCompressed]))
#        self.vidangeBuffer()
        return
        
    def addNewLongitudeFunction(self,startTime, endTime, f, nbpointsCompressed):
        self.longitudeResultArray=np.vstack((self.longitudeResultArray, [startTime,endTime,f[0],f[1],f[2], f[3], nbpointsCompressed]))       
        return     
        
    def majLastLatitudeFunction(self,startTime, endTime, f, nbpointsCompressed):
        self.latitudeResultArray[-1]=[startTime,endTime,f[0],f[1],f[2],f[3], nbpointsCompressed]
        return
        
    def majLastLongitudeFunction(self,startTime, endTime,f, nbpointsCompressed):
        self.longitudeResultArray[-1]=[startTime,endTime,f[0],f[1],f[2],f[3], nbpointsCompressed]
        return       
        
    #processus de recherche de fonction
    def processTrajectoryLongitude(self):
        # la fonction est lancée parce qu'on a suffisamment de points en entrée
        
        # 1ère étape: initialisation, si aucune fonction n'est définie
        if self.longitudeResultArray.size==0:
            tdeb,tfin,f,nbPoints=self.processFunctionResearch(self.timestamp, 
                                             self.longitude, self.epsilon,
                                             1000,self.degree)     
            self.addNewLongitudeFunction(tdeb, tfin, f,nbPoints) 
            
        # une fois calculée, on doit vérifier si les points fittent bien, qu'on a pas trop d'erreurs
        iiFirstErr,nbErr=self.controlErreurs(self.timestamp,
                                      self.longitude, 
                                      self.longitudeResultArray[-1], 
                                      self.epsilon)  
        # si à la vérification on constate que le nouveau point dépasse la tolérance
        if nbErr>0:
            #on ajuste la fonction sur le tronçon
            tdeb,tfin,f,nbErr2,iiFirstEr,nbPointsMaj=self.processFunctionResearchMaj(self.timestamp, 
                                             self.longitude,
                                             self.epsilon,
                                             self.longitudeResultArray[-1,0],
                                             self.degree)  
                               
            # si la nouvelle solution est meilleure (pas d'erreur), on ne boucle pas le tronçon
            # on se contente de mettre à jour les données
            if nbErr2==-1:
                self.majLastLongitudeFunction(tdeb, tfin, f,nbPointsMaj)
            # si on a des erreurs, on se calle sur la version initiale
            else:                  
                # on part à la recherche du nouveau tronçon    
                tdeb,tfin2,f,nbPoints =self.processFunctionResearch(self.timestamp, 
                                                 self.longitude,
                                                 self.epsilon,
                                                 tfin,1)
                                                                 
                self.addNewLongitudeFunction(min(tdeb, tfin), tfin2, f,nbPoints)                
                
        else:
            self.longitudeResultArray[-1,1]=self.timestamp[-1]    
            
        return 1     
        
    #processus de recherche de fonction
    def processTrajectoryLatitude(self):
#        tdeb=0
#        tfin=0
        # si on a suffisamment de points, on peu lancer un calcul
        if self.latitudeResultArray.size==0:
            tdeb,tfin,f,nbPoints=self.processFunctionResearch(self.timestamp, 
                                             self.latitude, self.epsilon,
                                             self.timestamp[0],self.degree)     
            self.addNewLatitudeFunction(tdeb, tfin, f,nbPoints) 
        
        # une fois calculée, on doit vérifier si les points fittent bien, qu'on a pas trop d'erreurs
        iiFirstErr,nbErr=self.controlErreurs(self.timestamp,
                                      self.latitude, 
                                      self.latitudeResultArray[-1], 
                                      self.epsilon)
        # si à la vérification on constate que le nouveau point dépasse la tolérance                              
        if nbErr>0:
            #on ajuste la fonction sur le tronçon
            tdeb,tfin,f,nbErr2,iiFirstEr,nbPointsMaj=self.processFunctionResearchMaj(self.timestamp, 
                                             self.latitude,
                                             self.epsilon,
                                             self.latitudeResultArray[-1,0],self.degree)                                             
             
            # si la nouvelle solution est meilleure, on ne boucle pas le tronçon
            # on se contente de mettre à jour les données
            if nbErr2==-1:
                self.majLastLatitudeFunction(tdeb, tfin, f,nbPointsMaj)
            # si on a des erreurs, on se calle sur la version initiale
            else: 
               
                # on part à la recherche du nouveau tronçon    
                tdeb,tfin2,f,nbPoints=self.processFunctionResearch(self.timestamp, 
                                                 self.latitude, 
                                                 self.epsilon,
                                                 tfin,1)                         
                self.addNewLatitudeFunction(min(tdeb, tfin), tfin2, f,nbPoints) 
            
            
        else:
            self.latitudeResultArray[-1,1]=self.timestamp[-1]          
               
        return 1   
        
    def controlErreurs(self,rawTime,rawValues, function2ctrl, epsilon):
        # on calcule les points obtenus par fonction pour les temps de ref
        f=[function2ctrl[4], function2ctrl[5]]
        valFromFunction=np.polyval(f,rawTime)
        
        # on calcul la distance de chacun de ces points vs raw data
        dist=np.fromiter(map(isub,rawValues,valFromFunction), np.float32)
        
        #on recherche l'indice des points qui ont une erreur au dessus d'epsilon   
        ii=np.where(np.logical_and(rawTime>=function2ctrl[0], abs(dist)>epsilon))
        
        if len(ii[0])>0:
#            print(np.array(ii)[0,:],len(ii[0]))
            return np.array(ii)[0,0],len(ii[0])
        else:
            return -1,-1    
        
    #processus de recherche de fonction
    def processFunctionResearch(self,input_time, input_position, epsilon_position, gap_max, degree):
        deg=degree        
        # nombre de points à notre disposition en input
        nb_item=input_time.size
        
        if nb_item<2:
            return
        # on prend les derniers points à disposition pour faire un prévision de 
        # trajectoire. Si on a suffisamment de points dans la dernière minutes, 
        # on fait avec. Sinon, on prend les deux derniers          
        ii=np.where(self.timestamp>self.timestamp[-1]-60)        
        
        nb_elements=max(2,len(ii[0]))
        
        element2process=[]
        
        if nb_elements>2: 
            element2process=ii[0]
        else:
            element2process=[len(input_time)-2, len(input_time)-1]            
        if nb_elements<10: 
            deg=1
            
        nbPointsCompressed=len(input_time[element2process])
        
        f=np.polyfit(input_time[element2process],
                      input_position[element2process],deg)
                      
        
        result_f=np.zeros((4-deg-1))

        result_f=np.concatenate((result_f,f))
                    
        return input_time[element2process[0]],input_time[element2process[-1]],result_f,nbPointsCompressed

    # cette fonction a pour rôle de réduire la taille des valeurs d'entrée, une fois qu'on a validé des 
    # les tronçons et qu'elles sont devenues inutiles à stocker
    def vidangeBuffer(self):  
        
        sizeLat=len(self.longitudeResultArray)
        sizeLon=len(self.longitudeResultArray)
        
        # si aucune fonction d'a été enregistrée on ne fait rien
        if sizeLat<=1 or sizeLon<=1:
            timestampLimite=self.timestamp[0]
        else:
            timestampLimite=min(self.longitudeResultArray[-1,0], self.latitudeResultArray[-1,0])-70
            ii2keep=np.where(self.timestamp>=timestampLimite)
                
            self.timestamp=self.timestamp[ii2keep[0]]
            self.latitude=self.latitude[ii2keep[0]]
            self.longitude=self.longitude[ii2keep[0]]
            
        return
        
        
    #processus de recherche de fonction
    def processFunctionResearchMaj(self,input_time, input_position, epsilon_position, timestampDebut, degree):
        
        ideb=np.where(input_time>=timestampDebut)        
    
        f=np.polyfit(input_time[ideb[0][0]:len(input_time)],
                      input_position[ideb[0][0]:len(input_time)],
                                     self.degree)
        
        result_f=np.zeros((4-degree-1))
        result_f=np.concatenate((result_f,f))                           
                                
        f2=[input_time[ideb[0][0]],input_time[-1],result_f[0], result_f[1],result_f[2], result_f[3]] 
        
        ii,nbError=self.controlErreurs(
        input_time[ideb[0][0]:len(input_time)],
        input_position[ideb[0][0]:len(input_time)],
        f2,
        epsilon_position
        )
        nbPointsCompressed=len(input_time[ideb[0][0]:len(input_time)])          
        
        return input_time[ideb[0][0]],input_time[-1],result_f,nbError,ii,nbPointsCompressed  #TODO : il faut aussi retourner les heures de début et fin de tronçon       
        
        


    def coordinates2DistanceInMeters(self,latA, lonA, latB, lonB ):
        earthRadius= 6378137
        res = earthRadius* math.acos(math.sin(math.radians(latA))*math.sin(math.radians(latB)) + math.cos(math.radians(latA))*math.cos(math.radians(latB))*math.cos(math.radians(lonA-lonB)))
        return res

    def coordinates2AzimuthInDegrees(self,latRef, lonRef, latTarget, lonTarget ):
        dx=lonTarget-lonRef
        dy=latTarget-latRef
        az = np.mod(360.0+90.0-math.degrees(math.atan2(dy,dx)),360.0);       
        return az
        
        
       
    
