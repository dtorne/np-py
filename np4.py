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
        varsInfoBits = []
        file = open("log.txt","w")
        file.write("\n clauses:" + str(clauses))
        for newVar in range(1,numVar+1):
            quantInfoVar = estimateQuantInfo(clauses, newVar, numVar)
            file.write("\n vars entropy: " + str(quantInfoVar))
            file.write("\n new var: " + str(newVar))
            file.write("\n num var: " + str(numVar))
            varsInfoBits.append((newVar, quantInfoVar))
        #We order descend, from the most restrictive variable to lowest, its expansion to 0 and 1 restricts most the problem
        #Bigest quantity of iformation or uncertainty to lowest once you assign the variable measured to the value it restricts every clause
        #We are concerned that when we expand the variable, it does not block any new assignment and size grows exponentially (x2),
        #so we want the var to interact with most clauses meaningfully to increases the chances to block or delete new assignments and minimize its expanded data
        #even at the expense of deleting clauses for the non restricting asignment branch. We value early deletion.
        varsInfoBits = sorted(varsInfoBits, key=lambda varTuple: varTuple[1], reverse=True)
        
        file.write("vars entropy:\n"+str(varsInfoBits))
        file.close()
        ind = 1
        for (ordVar,_) in varsInfoBits:
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
    estimateQuantInfo
    Calculates the quantity of information in bits of the problem given an assigned varE and its clauses P(I|VarE)
    VarE assumes the value in each clause that blocks it the most (impossibility both 1 and 0, but good to estimate if expand VarE)
    Probability(rough estimate) find a solution is all clauses value false Prob(SolveProb) = Mul I€(1..numClause) ProbClauseIFalse
    ProbClauseIFalse = 1 - ProbClauseIsTrue
    ProbClauseIsTrue (assaigning vars randomly independently each var in each clause p=1/2 x var) = (1/2) ^ len(clause)
    (treats all vars in a clause as different from other clauses)
    Finally quantInfo = log2(1/Prob(SolvProb)) = -log2(Prob(SolvProb)) =  Sum I€(1..numClause) -log2(ProbClauseIFalse)

    (every clause probability is calculated using a binary(0,1) random Variable prob(1/2) independent between clauses)
    That is probability to find a solution with random assignment Entropy = Sum i€(1..numClasses) = log(1/probClassFalse) 

"""
def estimateQuantInfo(clauses, varE, numVar):
    
    quantInfo = 0.0
    file = open("logE.txt","w")
    
    for clause in clauses:
        if varE in clause or (varE*-1) in clause:
            #Probability to make true the clause, is 1- prob(making clause false out of size clause random numbers 1 to numVar)
            #Probability match the clause
            probClauseIsTrue = (Decimal(1)/2) ** (len(clause)-1)
            file.write("\n len clause:" + str(len(clause)))
            #Probability make clause true
            probClauseIFalse = Decimal(1 - probClauseIsTrue)
            quantInfoCl = -math.log(probClauseIFalse , 2)
            quantInfo += quantInfoCl
            file.write("\n entropy Acc:" + str(quantInfo))
        else:
            probClauseIsTrue = (Decimal(1)/2) ** (len(clause))
            file.write("\n len clause:" + str(len(clause)))
            probClauseIFalse = Decimal(1 - probClauseIsTrue)
            quantInfoCl = -math.log(probClauseIFalse , 2)
            quantInfo += quantInfoCl
            file.write("\n entropy Acc:" + str(quantInfo))
    file.close()

    return quantInfo


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
    varMinimClause
    Select all the clauses that includes the var
"""
def varMinimClause(clauses, newVariable):  
    minClauses = set()
    newVars = set({newVariable,-newVariable})
    for clause in clauses:
        if newVars & clause != set():
            minClauses.add(clause)
    return minClauses
"""
    minimClauseAllVars
    Reduces the number of clauses to the minimum relevant where clauses that will not be used
    are not included     
    @param clauses Array of sets. Each set is a clause of type (int, int int...)
    @param level . How many variable to check. From 1 to level variables
    @param onlyNewVariables. if True, adds non level variables
    Requires level < newVariable.
"""
def minimClauseAllVars(clauses, level, onlyNewVariables = True):
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
        if isMinimum and ((not onlyNewVariables) or hasOneNew):
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
def updateState(clauses, idValue, level, numVar, initStatus = 3, onlyNewVariables = True):
       
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
        minimClauses = set()
        if onlyNewVariables:
            minimClauses = varMinimClause(clauses, newVariable)
        else:
            minimClauses = clauses
        
        #Check if 0 for newVariable was passing. If status 1 or 3
        if status % 2 == 1:
            idSet.add(-newVariable)
            passes0 = checkBoolean(minimClauses, instanceSet = idSet)
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
            passes1 = checkBoolean(minimClauses, instanceSet = idSet)
            idSet.remove(newVariable)
            if not passes1:
                #newVariable at 1 does not pass. Status then 1 or 0
                status = status % 2

        if status == 0:
            return -1
        else:
            finalStatus = status
    
    return finalStatus

"""
    expandLevel
    Creates a list of all  expanded ids when adding the newVariable. Based on stateCol binary value
    stateCol, if first bit = 1, says expand newVariable = 0- bit=0 not, segond bit = 1, says expand newVariable = 1 , bit=0 not.
"""
def expandLevel(idCol, stateCol, newVariable):
    result = []
    #We add newVar = 0
    if stateCol % 2 == 1:
        result.append(idCol)
    #We add newVar = 1
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
        #We want to check the initial expanded variables clauses only once, at the begining. Latter need only to check the extended clauses.
        onlyNewVariablesV = True
        if newVar == startLevel+1:
            onlyNewVariablesV = False

        reducedClause = minimClauseAllVars(clauses, newVar-1, onlyNewVariables = onlyNewVariablesV)       
        #Parse state with clauses
        walkerDF = walkerDF.rdd.map(lambda row: Row(id=row['id'], state=updateState(clauses = reducedClause, idValue = row['id'], level = newVar-1, numVar = numVar, onlyNewVariables = onlyNewVariablesV))).toDF().where(col('state') != -1)
        walkerDF = walkerDF.withColumn("id",explode(array(expandids(col("id"),col("state"),lit(newVar))))).withColumn("state",lit(3)).select(explode("id").alias("id"),"state").cache()
       
        if newVar % freqSave == 0:
            fileName = str(numVar)+"out"+ str(newVar)
            if reorderVars:
                fileName = fileName + "or"               
            walkerDF.select("id").write.save(fileName,format="csv")
        
    return walkerDF.select("id")

conf = SparkConf().setAppName("Boolean Solver")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
# 20 Vars - uf20-01.cnf
# 50 Vars - uf50-05.cnf
# 75 Vars - uf75-08.cnf Bug?
# 75 Vars - flat75-11.cnf

orderedVars = {}
graphDF = solveCNF("uf75-08.cnf", 24, inputPath = "75out24or", reorderVars = True, orderVars = orderedVars, freqSave = 1).cache()#, 15).cache()#36, "50output36").cache()

numSolutions = graphDF.count()
file = open("dictionary.txt","w")
file.write("Variables:\n")
file.write("Number solutions:" + str(numSolutions) +"\n")
file.write(json.dumps(orderedVars))
file.close()
    

