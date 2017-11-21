#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:03:33 2017

@author: david
"""

from pyspark.sql import Row
from pyspark.sql.types import LongType, ArrayType
from pyspark.sql.functions import array, explode, lit, col, udf
from pyspark.sql import *
from pyspark import SparkContext, SparkConf

from os import listdir
from os.path import isfile, join
from decimal import *
import sys
import re
import math
import json




def parseNPFunction(path):

    pattern = re.compile("[-]?\\d")
    
    numVar = 0
    numClauses = 0
    clauses = set()
    file = open(path, 'r') 
    for line in file:
        lineA = line.split()
        if len(lineA) == 0:
            continue
        
        if lineA[0] == "p":
            numVar = int(lineA[2])
            numClauses = int(lineA[3])
            
        elif pattern.match(lineA[0]):
            clause = set(map(lambda x : int(x), lineA[:-1]))
            if len(clause) > 0:
                clauses.add(frozenset(clause))
    
    numClauses = len(clauses)
    return (numVar, numClauses, clauses)

"""
    heuristicVarOrder
    Based on a simple heuristic inference using one variable information gain measure, infers an initial estimate of order
    of the most blocking variables. It reorders the clauses names based on that order.
    If the order is given in varDict, it uses directly that order.
    Outputs (clauses, varsOrdered) where clauses substitute the previous numbers, by the order assignment in varsOrdered
    e.g. (clauses={1,4,5}, {4,7,8}) based on varsOrdered={ 0:0 , 1:7 , 2:4 , 3:3 , 4:5 , 5:8 , 6:6 , 7:2 , 8:3 }
    is clauses({7,5,8}, {5,2,3}) Leftmost varOrdered (Lower keys) are the most blocking aka 1,2,3...
    It modifies varDict input object if given explicitelly empty.

"""
def heuristicVarOrder(clauses, numVar, varDict={} ):
    
    if varDict == {}:
        varsEntropy = []
        file = open("log.txt","w")
        file.write("\n clauses:" + str(clauses))
        for newVar in range(1,numVar+1):
            entropyVar = estimateEntropy(clauses, newVar, numVar)
            file.write("\n vars entropy: " + str(entropyVar))
            file.write("\n new var: " + str(newVar))
            file.write("\n num var: " + str(numVar))
            varsEntropy.append((newVar, entropyVar))
        varsEntropy = sorted(varsEntropy, key=lambda varTuple: varTuple[1], reverse=True)
        
        file.write("vars entropy:\n"+str(varsEntropy))
        file.close()
        ind = 1
        for (ordVar,_) in varsEntropy:
            varDict[ordVar] = ind
            ind += 1

    renamedClauses = set()
    for clause in clauses:
        newClause = set()
        for varI in clause:
            if varI > 0:
                newClause.add(varDict[varI])
            else:
                newClause.add(-varDict[-varI])
        renamedClauses.add(frozenset(newClause))

    return renamedClauses

"""
    estimateEntropy
    Calculates the entropy estimate given a var and its clauses
"""
def estimateEntropy(clauses, varE, numVar):
    #Probability all clauses values true (every clause random Variable is independend of each other close in this estimation)
    entropy = 0.0#
    file = open("logE.txt","w")
    
    for clause in clauses:
        if varE in clause or (varE*-1) in clause:
            #Probability to make true the clause, is 1- prob(making clause false out of size clause random numbers 1 to numVar)
            #Probability match the clause
            prob = (Decimal(1)/2) ** (len(clause)-1)
            file.write("\n len clause:" + str(len(clause)))
            #Probability make clause true
            prob = Decimal(1 - prob)
            entropyClause = math.log(Decimal(1) / prob , 2)
            entropy += entropyClause
            file.write("\n entropy Acc:" + str(entropy))
        else:
            prob = (Decimal(1)/2) ** (len(clause))
            file.write("\n len clause:" + str(len(clause)))
            prob = Decimal(1 - prob)
            entropyClause = math.log(Decimal(1) / prob , 2)
            entropy += entropyClause
            file.write("\n entropy Acc:" + str(entropy))
    file.close()

    return entropy


"""
    parseInput
    Reads as input the output path of a previous run. Paths of the form numVars+output+currentLevel
"""
def parseInput(path):
    pattern = re.compile(".*csv$")
    data = []    
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for fileName in files:
        if pattern.match(fileName):
            file = open(path + "/" + fileName, 'r')
            for line in file:
                data.append(long(line))     
    return data


"""
    minimclause
    Reduces the number of clauses to the minimum relevant where newVar appears (rest have been already checked)
    @param clauses Array of sets. Each set is a clause of type (int, int int...)
    @param level . How many variable to check. From 1 to level + the new variable
    @param newVariable. the new variable whose valid clause add to the existing
    Requires level < newVariable.
"""
def minimClause(clauses, level, newVariable, onlynewVariable = True):
    varsFilter = set(range(1,level+1))
    varsFilter.add(newVariable)
    minClauses = set()
    for clause in clauses:
        isMinimum = True
        hasnewVariable = False
        for item in clause:
            if abs(item) not in varsFilter:
                isMinimum = False
                break
            elif abs(item) == newVariable:
                hasnewVariable = True
        if isMinimum and ((not onlynewVariable) or hasnewVariable):
            minClauses.add(clause)
    return minClauses

"""
    minimClauseAllVars
    Reduces the number of clauses to the minimum relevant where clauses that will not be used
    are not included     
    @param clauses Array of sets. Each set is a clause of type (int, int int...)
    @param level . How many variable to check. From 1 to level variables
    @param onlynewVariables. if True, adds non level variables
    Requires level < newVariable.
"""
def minimClauseAllVars(clauses, level, onlynewVariables = True):
    varsFilter = set(range(1,level+1))
    minClauses = set()
    for clause in clauses:
        isMinimum = False
        hasOneNew = False
        for item in clause:
            if abs(item) in varsFilter:
                isMinimum = True
            else:
                hasOneNew = True
        if isMinimum and ((not onlynewVariables) or hasOneNew):
                minClauses.add(frozenset(clause))
               
    return minClauses



"""
    checkBoolean
    returns True if exist solution, False otherwise
"""
def checkBoolean(clauses, instanceSet):
    for clause in clauses:
        isSolution = False
        for item in clause:
            if item not in instanceSet:
                isSolution = True
        if not isSolution:
            return isSolution
    return True

"""
    updateState
    Calculates if this instantiation passes the closes.
    Returns status. 
    -1- delete. passes neither 0 nor 1 newVar instance for this id instance
    1- 01b passes newVar 0 
    2- 10b passes newVar 1
    3- 11b passes newVar 0 and 1
    Accpetance condition level < newVariable. Otherwise unexpected behaviour
"""
def updateState(clauses, idValue, level, numVar, initStatus = 3):
       
    idSet = set()
    for i in range(0,level):
        if (idValue >> i) % 2 == 1:
            idSet.add(i+1)
        else:
            idSet.add(-(i+1))
    
    finalStatus = 0
    #Since it goes from final to newVar = level+1, updates status with the final case level+1
    #The rest of variables are used to check if it, the instance, can be eleminated. status = -1
    for newVariable in range(numVar,level,-1):
        status = initStatus
        #Check if 0 for newVariable was passing. If status 1 or 3
        if status % 2 == 1:
            idSet.add(-newVariable)
            passes0 = checkBoolean(clauses, instanceSet = idSet)
            idSet.remove(-newVariable)
            if not passes0:
                #newVariable at 0 does not pass. Status then 2 or 0
                status = status - 1
            #For the rest of variables than level+1, we just want to see if it can be eliminated with both passes0 and passes1
            #So we continue directly to next iteration if we know in this newVariable there is solution
            elif newVariable != level+1:
                continue
  
        #Check if 1 for newVariable was passing. Status 2 or 3
        if (status>>1) % 2 == 1:
            idSet.add(newVariable)
            passes1 = checkBoolean(clauses, instanceSet = idSet)
            if not passes1:
                #newVariable at 1 does not pass. Status then 1 or 0
                status = status % 2
            idSet.remove(newVariable)
        if status == 0:
            return -1
        else:
            finalStatus = status
    
    return finalStatus


def expandLevel(idCol,stateCol,newVariable):
    result = []
    #We add newVar = 0
    if stateCol % 2 == 1:
        result.append(idCol)
    if (stateCol>>1) % 2 == 1:
        result.append(2**(newVariable-1)+idCol)
    return result

"""
    solveCNF
    Input path must containt solutions that correspond to the number of variables expanded = startLevel
    orderVars passed object returns modified if reorderVars = True, and  orderVars is given empty
"""
def solveCNF(path="", startLevel=10, inputPath = "", reorderVars = False, orderVars = {}, freqSave = -1):
    
    if path == "":
        path = path

    expandids = udf(expandLevel, ArrayType(LongType()))
    (numVar, numClauses, clauses) = parseNPFunction(path)
    
    if freqSave == -1:
        freqSave = numVar
    
    if reorderVars:
        #Modifies also orderVars. Dictionary
        clauses = heuristicVarOrder(clauses, numVar, orderVars)
     
    
    initData = []
    if inputPath == "":
        initData = [i for i in range(0,2**(startLevel))]
    else:
        initData = parseInput(inputPath)
        
    initRDD = sc.parallelize(initData)
    initDF = initRDD.map(lambda x: Row(id=long(x), state=3))
        
    walkerDF = sqlContext.createDataFrame(initDF).cache()
    
    for newVar in range(startLevel+1, numVar+1):
        if newVar == startLevel+1:
            reducedClause = minimClauseAllVars(clauses, newVar-1, onlynewVariables = False)
        else:
            reducedClause = minimClauseAllVars(clauses, newVar-1)
        
        #Parse state with clauses
        walkerDF = walkerDF.rdd.map(lambda row: Row(id=row['id'], state=updateState(clauses = reducedClause, idValue = row['id'], level = newVar-1, numVar = numVar))).toDF().where(col('state') != -1)
        walkerDF = walkerDF.withColumn("id",explode(array(expandids(col("id"),col("state"),lit(newVar))))).withColumn("state",lit(3)).select(explode("id").alias("id"),"state").cache()
       
        if newVar % freqSave == 0:
            walkerDF.select("id").write.save(str(numVar)+"output"+ str(newVar),format="csv")
        
    #return (walkerDF.select("id"), orderVars)
    #TODO return orderVars or print
    return walkerDF.select("id")

conf = SparkConf().setAppName("Boolean Solver")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
# 20 Vars - uf20-01.cnf
# 50 Vars - uf50-05.cnf
# 75 Vars - uf75-04.cnf

orderedVars = {}
graphDF = solveCNF("uf50-05.cnf", 41, inputPath = "50output41", reorderVars = True, orderVars = orderedVars, freqSave = 1).cache()#, 15).cache()#36, "50output36").cache()

numSolutions = graphDF.count()
file = open("dictionary.txt","w")
file.write("Variables:\n")
file.write("Number solutions:" + numSolutions +"\n")
file.write(json.dumps(orderedVars))
file.close()
    

